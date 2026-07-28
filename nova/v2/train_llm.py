"""
Nova V2 - Train a Fresh LLM from Scratch
===========================================
Step-by-step: raw text -> V2 learns patterns -> save -> query later.

No PyTorch. No Transformers. No backpropagation.

Usage:
    python nova/v2/train_llm.py --data tiny_stories --epochs 3
    
After training, model saved to 'v2_llm_model/' folder.
Then query:
    from nova.v2.generator import TextGenerator
    gen = TextGenerator.load("v2_llm_model")
    print(gen.query("Once upon a time"))
"""
import sys, os, math, time, random, json
from collections import Counter
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from nova.v2.fingerprint import djb2_hash, StructuralFingerprint, FingerprintParser
from nova.v2.memory import RoleVocabulary
from nova.v2.rules import RuleEngine, create_default_rules
from nova.v2.learning import LearningEngine
from nova.v2.generator import ReverseHashTable, TextGenerator


def load_text(data_source: str, max_tokens: int = 100000) -> List[str]:
    """Load text from file, HF dataset, or synthetic."""
    tokens = []
    
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
                if text and text.strip():
                    tokens.extend(text.strip().split()[:max_tokens - len(tokens)])
            print(f"  Loaded {len(tokens)} tokens")
        except ImportError:
            print("  datasets not installed. Using synthetic data.")
            words = ["the","cat","dog","bird","ran","sat","flew","a","on","mat","sun"]
            tokens = [random.choice(words) for _ in range(max_tokens)]
    
    elif os.path.exists(data_source):
        with open(data_source, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = text.strip().split()[:max_tokens]
        print(f"  Loaded {len(tokens)} tokens from {data_source}")
    else:
        print(f"  Using synthetic data")
        words = ["the","cat","dog","bird","ran","sat","flew","a","on","mat","sun"]
        tokens = [random.choice(words) for _ in range(max_tokens)]
    
    return tokens


def train_v2_llm(
    data_source: str = "tiny_stories",
    epochs: int = 3,
    context_size: int = 3,
    max_tokens: int = 50000,
    min_freq: int = 1,
    save_path: str = "v2_llm_model",
) -> TextGenerator:
    """
    Train a complete V2 LLM. Model saved to save_path.
    
    Returns:
        TextGenerator — ready to query with .query(prompt)
    """
    print("=" * 70)
    print("NOVA V2 - Training a Fresh LLM")
    print("No PyTorch. No Transformers. No backpropagation.")
    print("=" * 70)
    
    print("\n[1/5] LOADING DATA")
    all_tokens = load_text(data_source, max_tokens)
    freq = Counter(all_tokens)
    tokens = [t for t in all_tokens if freq[t] >= min_freq]
    unique_tokens = set(tokens)
    
    print("\n[2/5] BUILDING REVERSE HASH TABLE")
    reverse = ReverseHashTable()
    for t in unique_tokens:
        reverse.add(t, freq[t])
    print(f"  {reverse.size} hashes, {reverse.total_tokens} tokens")
    
    print("\n[3/5] INITIALIZING V2")
    roles = RoleVocabulary()
    parser = FingerprintParser()
    parser.fit(tokens)
    engine = RuleEngine()
    for rule in create_default_rules(roles):
        engine.add_rule(rule)
    learner = LearningEngine(engine, roles)
    print(f"  Vocab: {len(unique_tokens)} tokens")
    
    print("\n[4/5] TRAINING")
    total_pairs = 0
    start = time.time()
    
    for epoch in range(epochs):
        epoch_correct = 0
        for i in range(context_size, len(tokens)):
            ctx = tokens[max(0, i - context_size):i]
            stats = learner.train_step(ctx, tokens[i], parser)
            if stats['correct']:
                epoch_correct += 1
            total_pairs += 1
            if total_pairs % 25000 == 0:
                elapsed = time.time() - start
                print(f"  Ep {epoch+1}/{epochs} | {total_pairs}/{len(tokens)*epochs} "
                      f"| Acc: {stats['accuracy']:.0%} | {total_pairs/max(elapsed,0.1):.0f} tok/s")
        
        acc = epoch_correct / max(len(tokens) - context_size, 1)
        print(f"  Epoch {epoch+1} done: accuracy = {acc:.1%}")
    
    elapsed = time.time() - start
    print(f"\n  Trained: {total_pairs} pairs in {elapsed:.1f}s ({total_pairs/max(elapsed,0.1):.0f} tok/s)")
    
    print("\n[5/5] SAVING MODEL")
    gen = TextGenerator(learner, reverse, parser)
    gen.save(save_path)
    
    # Test
    print("\n  Test generation:")
    for prompt in ["the", "a"]:
        out = gen.generate(prompt, max_tokens=15)
        print(f"    '{prompt}' -> '{out}'")
    
    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Model saved to: '{save_path}/'")
    print(f"  Query it: gen = TextGenerator.load('{save_path}')")
    print(f"  Example:   print(gen.query('Once upon a time'))")
    print(f"{'=' * 70}")
    
    return gen


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Train Nova V2 LLM")
    p.add_argument('--data', default='tiny_stories')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--context', type=int, default=3)
    p.add_argument('--tokens', type=int, default=50000)
    p.add_argument('--min-freq', type=int, default=1)
    p.add_argument('--save', default='v2_llm_model')
    args = p.parse_args()
    
    train_v2_llm(
        data_source=args.data,
        epochs=args.epochs,
        context_size=args.context,
        max_tokens=args.tokens,
        min_freq=args.min_freq,
        save_path=args.save,
    )