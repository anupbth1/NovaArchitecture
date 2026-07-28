"""
Nova V2 - Train a Fresh LLM from Scratch
===========================================
Step-by-step: raw text -> V2 learns patterns -> generate new text.

This is a COMPLETE training pipeline using V2 architecture.
No PyTorch. No Transformers. No backpropagation.

Process:
1. Load text data (any .txt file or HF dataset)
2. Parse tokens -> fingerprints
3. Build reverse hash table (replaces softmax vocabulary)
4. Train: observe context->target pairs, store in context map
5. Generate: predict next hash -> reverse lookup -> text

Complexity: O(T * M) where T = tokens, M = context window
Memory: ~100MB for 1M tokens (vs GBs for Transformer training)

Usage:
    python nova/v2/train_llm.py --data tiny_stories --epochs 3
    python nova/v2/train_llm.py --data my_text.txt --epochs 5
    python nova/v2/train_llm.py --data hf://wikitext --epochs 2
"""
import sys, os, math, time, random, json
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from nova.v2.fingerprint import djb2_hash, StructuralFingerprint, FingerprintParser
from nova.v2.memory import WorkingMemory, RoleVocabulary
from nova.v2.rules import RuleEngine, Pattern, Action, Rule, create_default_rules
from nova.v2.learning import LearningEngine
from nova.v2.generator import ReverseHashTable, TextGenerator


def load_text(data_source: str, max_tokens: int = 100000) -> List[str]:
    """
    Load text from various sources.
    
    Supported:
    - "tiny_stories" -> HF TinyStories dataset (first 50K tokens)
    - "wikitext" -> HF WikiText-2 (first 50K tokens)
    - "hf://dataset:split:column" -> Any HF dataset
    - "path/to/file.txt" -> Local text file
    """
    tokens = []
    
    # HuggingFace datasets
    if data_source in ["tiny_stories", "wikitext"] or data_source.startswith("hf://"):
        try:
            from datasets import load_dataset
            if data_source == "tiny_stories":
                ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
                col = "text"
            elif data_source == "wikitext":
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
                col = "text"
            else:
                parts = data_source.replace("hf://", "").split(":")
                ds = load_dataset(parts[0], split=parts[1] if len(parts)>1 else "train", streaming=True)
                col = parts[2] if len(parts)>2 else "text"
            
            print(f"  Loading from HuggingFace: {data_source}")
            for i, example in enumerate(ds):
                if len(tokens) >= max_tokens:
                    break
                text = example[col]
                if text.strip():
                    tokens.extend(text.strip().split()[:max_tokens - len(tokens)])
            print(f"  Loaded {len(tokens)} tokens")
            
        except ImportError:
            print("  datasets not installed. Install: pip install datasets")
            print("  Using synthetic data instead.")
            tokens = ["the", "cat", "sat", "on", "the", "mat"] * (max_tokens // 6)
    
    # Local file
    elif os.path.exists(data_source):
        with open(data_source, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = text.strip().split()[:max_tokens]
        print(f"  Loaded {len(tokens)} tokens from {data_source}")
    
    # Synthetic data
    else:
        print(f"  Using synthetic data (no real dataset found)")
        words = ["the", "cat", "dog", "bird", "ran", "sat", "flew", "sang",
                 "a", "quickly", "beautifully", "home", "away", "on", "mat",
                 "sun", "moon", "star", "sky", "tree", "flower", "river"]
        tokens = [random.choice(words) for _ in range(max_tokens)]
    
    return tokens


def train_v2_llm(
    data_source: str = "tiny_stories",
    epochs: int = 3,
    context_size: int = 3,
    max_tokens: int = 50000,
    min_freq: int = 2,
) -> Tuple[LearningEngine, ReverseHashTable, FingerprintParser, TextGenerator]:
    """
    Train a complete V2 LLM from scratch.
    
    Args:
        data_source: Text source (file, HF dataset, or synthetic)
        epochs: Number of training passes over data
        context_size: How many previous tokens to use as context
        max_tokens: Maximum tokens to train on
        min_freq: Minimum token frequency to include
        
    Returns:
        (learner, reverse_table, parser, generator) - ready to use
    """
    print("=" * 70)
    print("NOVA V2 - Training a Fresh LLM")
    print("No PyTorch. No Transformers. No backpropagation.")
    print("=" * 70)
    
    # Step 1: Load and filter data
    print("\n[1/6] LOADING DATA")
    all_tokens = load_text(data_source, max_tokens)
    
    # Filter by frequency
    freq = Counter(all_tokens)
    tokens = [t for t in all_tokens if freq[t] >= min_freq]
    print(f"  After frequency filter: {len(tokens)} tokens ({len(freq)} unique)")
    
    # Step 2: Build reverse hash table (replaces softmax vocabulary)
    print("\n[2/6] BUILDING REVERSE HASH TABLE")
    reverse = ReverseHashTable()
    unique_tokens = set(tokens)
    for t in unique_tokens:
        reverse.add(t, freq[t])
    print(f"  Reverse table: {reverse.size} hashes, {reverse.total_tokens} tokens")
    print(f"  Memory: ~{reverse.size * 32} bytes (vs {len(unique_tokens) * 2048 * 4} for embeddings)")
    
    # Step 3: Initialize V2 components
    print("\n[3/6] INITIALIZING V2 ARCHITECTURE")
    roles = RoleVocabulary()
    parser = FingerprintParser()
    parser.fit(tokens)
    
    engine = RuleEngine()
    for rule in create_default_rules(roles):
        engine.add_rule(rule)
    
    learner = LearningEngine(engine, roles)
    print(f"  Rules: {len(engine)}")
    print(f"  Vocabulary: {len(unique_tokens)} tokens")
    
    # Step 4: Training loop
    print("\n[4/6] TRAINING")
    total_pairs = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_correct = 0
        epoch_total = 0
        
        for i in range(context_size, len(tokens)):
            ctx = tokens[max(0, i - context_size):i]
            target = tokens[i]
            
            stats = learner.train_step(ctx, target, parser)
            
            if stats['correct']:
                epoch_correct += 1
            epoch_total += 1
            total_pairs += 1
            
            if total_pairs % 5000 == 0:
                elapsed = time.time() - start_time
                tok_s = total_pairs / max(elapsed, 0.1)
                print(f"  Epoch {epoch+1}/{epochs} | "
                      f"Tokens: {total_pairs}/{len(tokens)*epochs} | "
                      f"Acc: {stats['accuracy']:.0%} | "
                      f"Rules: {len(engine)} | "
                      f"Speed: {tok_s:.0f} tok/s")
        
        epoch_acc = epoch_correct / max(epoch_total, 1)
        print(f"  Epoch {epoch+1} complete: accuracy = {epoch_acc:.1%}")
    
    elapsed = time.time() - start_time
    print(f"\n  Training complete: {total_pairs} pairs in {elapsed:.1f}s")
    print(f"  Average speed: {total_pairs/max(elapsed,0.1):.0f} tokens/second")
    
    # Step 5: Create generator
    print("\n[5/6] CREATING GENERATOR")
    generator = TextGenerator(learner, reverse, parser)
    
    # Step 6: Test generation
    print("\n[6/6] TEST GENERATION")
    test_prompts = ["the", "a", "once", "there"]
    for prompt in test_prompts:
        output = generator.generate(prompt, max_tokens=15, temperature=0.7)
        print(f"  '{prompt}' -> '{output}'")
    
    # Summary
    memory_bytes = reverse.size * 32 + len(learner._context_map) * 32
    embedding_bytes = len(unique_tokens) * 2048 * 4
    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Tokens trained: {total_pairs}")
    print(f"  Memorized patterns: {len(learner._context_map)}")
    print(f"  Rules: {len(engine)}")
    print(f"  Memory: {memory_bytes/1024:.1f} KB (vs {embedding_bytes/1024/1024:.0f} MB for embeddings)")
    print(f"  Speed: {total_pairs/max(elapsed,0.1):.0f} tok/s (CPU)")
    print(f"\nV2 LLM ready! Use generator.generate(prompt) for inference.")
    print(f"{'=' * 70}")
    
    return learner, reverse, parser, generator


def demo():
    """Quick demo with synthetic data."""
    train_v2_llm(
        data_source="synthetic",
        epochs=2,
        context_size=3,
        max_tokens=5000,
        min_freq=1,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train Nova V2 LLM from scratch")
    parser.add_argument('--data', default='tiny_stories',
                       help='Data source: file.txt, tiny_stories, wikitext, hf://...')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--context', type=int, default=3, help='Context window size')
    parser.add_argument('--tokens', type=int, default=50000, help='Max tokens to train on')
    parser.add_argument('--min-freq', type=int, default=2, help='Min token frequency')
    
    args = parser.parse_args()
    train_v2_llm(
        data_source=args.data,
        epochs=args.epochs,
        context_size=args.context,
        max_tokens=args.tokens,
        min_freq=args.min_freq,
    )