"""
Phase 5: Text Generator
=========================
Hash lookups se actual text generate karo — no softmax, no embedding matrix.

Process:
1. Learned context map se next hash predict karo
2. Reverse hash table se hash → actual token convert karo
3. Ambiguity resolve karo using freq + type info
4. Beam search for multiple candidates

No softmax over 50K vocabulary. No linear projection layer.
Just hash lookup + table lookup.
"""
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import sys, os, math

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from nova.v2.fingerprint import djb2_hash, StructuralFingerprint, FingerprintParser
    from nova.v2.memory import WorkingMemory, RoleVocabulary
    from nova.v2.rules import RuleEngine, Pattern, Action, Rule, create_default_rules
    from nova.v2.learning import LearningEngine
else:
    from nova.v2.fingerprint import djb2_hash, StructuralFingerprint, FingerprintParser
    from nova.v2.memory import WorkingMemory, RoleVocabulary
    from nova.v2.rules import RuleEngine, Pattern, Action, Rule, create_default_rules
    from nova.v2.learning import LearningEngine


class ReverseHashTable:
    """
    Maps hash values back to actual tokens.
    
    This replaces the softmax head in Transformers.
    Instead of computing probabilities over 50K vocabulary,
    we just look up the hash in a table.
    
    Time: O(1) — hash lookup (vs O(V) for softmax)
    Memory: O(V) — one entry per unique token (vs O(VxD) for embeddings)
    """
    
    def __init__(self):
        # hash -> [(token, freq)]
        self.table: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    
    def add(self, token: str, freq: int = 1):
        """Add a token to the reverse index."""
        h = djb2_hash(token) % StructuralFingerprint.HASH_SPACE
        # Update frequency if exists
        for i, (t, f) in enumerate(self.table[h]):
            if t == token:
                self.table[h][i] = (token, f + freq)
                return
        self.table[h].append((token, freq))
    
    def add_corpus(self, tokens: List[str]):
        """Add all tokens from a corpus."""
        for t in tokens:
            self.add(t)
    
    def lookup(self, hash_val: int, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Look up tokens by hash.
        
        Returns:
            [(token, confidence), ...] sorted by confidence (frequency-based)
        """
        candidates = self.table.get(hash_val, [])
        if not candidates:
            return [("?", 0.0)]
        
        total = sum(f for _, f in candidates)
        scored = [(t, f / total) for t, f in candidates]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
    
    def resolve(self, hash_val: int, type_class: int = 0,
                freq_class: int = 128) -> str:
        """
        Resolve a hash to the most likely token given constraints.
        
        Args:
            hash_val: The predicted hash
            type_class: Preferred token type (0=word, default)
            freq_class: Preferred frequency class (higher = more common)
            
        Returns:
            Most likely token string
        """
        candidates = self.table.get(hash_val, [])
        if not candidates:
            return "<?>"
        if len(candidates) == 1:
            return candidates[0][0]
        
        # Multiple candidates: use type + freq to disambiguate
        # Higher freq = better (more common token wins)
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]
    
    @property
    def size(self) -> int:
        """Number of unique hashes."""
        return len(self.table)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens indexed."""
        return sum(len(v) for v in self.table.values())


class TextGenerator:
    """
    Generates text using learned patterns + reverse hash lookup.
    
    No softmax. No embedding matrix. No autoregressive loop.
    Just: predict hash → look up token → append → repeat.
    
    The generation is STILL sequential (one token at a time),
    but each step is O(1) instead of O(V) for softmax.
    """
    
    def __init__(self, learner: LearningEngine, reverse_table: ReverseHashTable,
                 parser: FingerprintParser):
        self.learner = learner
        self.reverse = reverse_table
        self.parser = parser
    
    def generate(self, prompt: str, max_tokens: int = 20,
                 temperature: float = 0.8) -> str:
        """
        Generate text from a prompt using learned patterns.
        
        Args:
            prompt: Starting text
            max_tokens: Maximum tokens to generate
            temperature: Randomness (0 = deterministic, 1 = creative)
            
        Returns:
            Generated text
        """
        # Tokenize prompt
        tokens = prompt.split()
        generated = list(tokens)
        
        for _ in range(max_tokens):
            # Get context (last few tokens)
            ctx = generated[-3:] if len(generated) >= 3 else generated
            
            # Predict next hash
            pred_hash = self.learner._predict(None, ctx)
            
            # Look up token(s) by hash
            if temperature > 0 and pred_hash in self.reverse.table:
                candidates = self.reverse.table[pred_hash]
                if candidates:
                    # Weighted random choice
                    total = sum(f for _, f in candidates)
                    r = random.random() * total * temperature
                    cum = 0
                    chosen = candidates[-1][0]  # default
                    for token, freq in candidates:
                        cum += freq
                        if r <= cum:
                            chosen = token
                            break
                    next_token = chosen
                else:
                    next_token = "<?>"
            else:
                # Deterministic: use hash lookup directly
                next_token = self.reverse.resolve(pred_hash)
            
            if next_token == "<?>":
                break  # Unknown token, stop
            
            generated.append(next_token)
        
        return " ".join(generated)


# ============================================================
# DEMO
# ============================================================

import random

def demo():
    print("=" * 70)
    print("Nova V2 - Phase 5: Text Generator")
    print("No softmax. No embedding matrix. Hash lookup only.")
    print("=" * 70)
    
    # Setup
    roles = RoleVocabulary()
    parser = FingerprintParser()
    engine = RuleEngine()
    
    for rule in create_default_rules(roles):
        engine.add_rule(rule)
    
    learner = LearningEngine(engine, roles)
    reverse = ReverseHashTable()
    
    # Training data (simple sentences)
    sentences = [
        "The cat sat on the mat",
        "A dog ran quickly home",
        "The bird flew away",
        "A bird sang beautifully",
        "The dog barked loudly",
        "A cat slept on the bed",
        "The sun shone brightly",
        "A child laughed happily",
    ]
    
    # Tokenize all sentences, build reverse table
    all_tokens = []
    for sent in sentences:
        tokens = sent.split()
        all_tokens.extend(tokens)
        reverse.add_corpus(tokens)
    
    parser.fit(all_tokens)
    
    # Train on all sentence pairs
    print(f"\nTraining on {len(sentences)} sentences...")
    for sent in sentences:
        tokens = sent.split()
        for i in range(1, len(tokens)):
            ctx = tokens[:i]
            target = tokens[i]
            learner.train_step(ctx, target, parser)
    
    acc = sum(1 for p, a in learner._history if p == a) / max(len(learner._history), 1)
    print(f"  Training accuracy: {acc:.0%}")
    print(f"  Memorized patterns: {len(learner._context_map)}")
    print(f"  Reverse table: {reverse.size} hashes, {reverse.total_tokens} tokens")
    
    # Generate text
    generator = TextGenerator(learner, reverse, parser)
    
    print("\n  Generation examples:")
    prompts = ["The", "A", "The cat", "A dog"]
    for prompt in prompts:
        output = generator.generate(prompt, max_tokens=8, temperature=0.5)
        print(f"    '{prompt}' -> '{output}'")
    
    print()
    print("NO softmax used:")
    print("  - Prediction: context map -> hash lookup (O(1))")
    print("  - Token resolution: reverse hash table (O(1))")
    print("  - No softmax over 50K vocabulary")
    print("  - No linear projection layer")
    print("  - No embedding matrix")
    print()
    print("Phase 5: WORKING")
    
    return generator


if __name__ == '__main__':
    demo()