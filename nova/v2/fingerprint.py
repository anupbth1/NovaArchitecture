"""
Phase 1: Structural Fingerprint
================================
Token ko bina embedding matrix ke represent karna.

Rule: No embedding matrix. No vector lookup. No d_model.
Only: 4 integers per token → (hash, freq_class, type_class, pos)

hash       = DJB2 hash of token string (mod 2²⁰)
freq_class = frequency quantile (1-256)
type_class = token type (0=word, 1=number, 2=punct, 3=special)
pos        = position in sequence

No torch.nn.Parameter. No Linear layers. Just integer operations.
"""
import math
from typing import List, Tuple, Optional


class StructuralFingerprint:
    """
    A token represented as structural features — no vectors.
    
    Fields:
        hash: int       — DJB2 hash of the raw token string (0 .. HASH_SPACE-1)
        freq_class: int — how common is this token? 1=rare, 256=very common
        type_class: int — word(0), number(1), punctuation(2), special(3)
        pos: int        — absolute position in the sequence
    
    Memory: 4 × 4 bytes = 16 bytes per token (vs 2048×4 = 8192 bytes for embedding)
    """
    
    HASH_SPACE = 2 ** 20  # ~1M possible hash values
    
    def __init__(self, hash: int, freq_class: int, type_class: int, pos: int):
        self.hash = hash % self.HASH_SPACE
        self.freq_class = min(255, max(1, freq_class))
        self.type_class = type_class & 3  # 0-3
        self.pos = pos
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return as plain tuple (no objects needed at runtime)."""
        return (self.hash, self.freq_class, self.type_class, self.pos)
    
    def __repr__(self) -> str:
        return f"FP(h={self.hash}, f={self.freq_class}, t={self.type_class}, p={self.pos})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, StructuralFingerprint):
            return False
        return (self.hash == other.hash and 
                self.freq_class == other.freq_class and
                self.type_class == other.type_class and
                self.pos == other.pos)
    
    def __hash__(self) -> int:
        return hash(self.to_tuple())
    
    def distance(self, other: 'StructuralFingerprint') -> int:
        """Structural distance between two fingerprints (NOT cosine similarity)."""
        d = 0
        d += 0 if self.hash == other.hash else 1
        d += abs(self.freq_class - other.freq_class) // 16  # 0-16 range
        d += 0 if self.type_class == other.type_class else 1
        d += abs(self.pos - other.pos)
        return d


# ============================================================
# HASHING (No embedding matrix)
# ============================================================

def djb2_hash(token: str) -> int:
    """
    DJB2 hash — deterministic, fast, no learned parameters.
    
    This replaces the embedding matrix lookup.
    Same token = same hash (deterministic).
    Different tokens → likely different hashes (good distribution).
    """
    h = 5381
    for c in token:
        h = ((h << 5) + h) + ord(c)
    return h & 0xFFFFFFFF  # 32-bit


def type_of_token(token: str) -> int:
    """Classify token type. Returns 0-3."""
    if not token:
        return 3  # special
    if token.isdigit():
        return 1  # number
    if all(c in '.,!?;:\'"-()[]{}' for c in token):
        return 2  # punctuation
    return 0  # word


# ============================================================
# FREQUENCY ESTIMATOR
# ============================================================

class FrequencyTracker:
    """
    Tracks token frequency across a corpus.
    Used to compute freq_class (how rare/common is a token).
    
    No parameters. Just counting.
    """
    def __init__(self):
        self.counts: dict = {}  # hash → count
        self.total = 0
    
    def observe(self, token: str):
        h = djb2_hash(token)
        self.counts[h] = self.counts.get(h, 0) + 1
        self.total += 1
    
    def observe_batch(self, tokens: List[str]):
        for t in tokens:
            self.observe(t)
    
    def freq_class(self, token: str, num_classes: int = 256) -> int:
        """Return frequency quantile (1 = rarest, 256 = most common)."""
        h = djb2_hash(token)
        count = self.counts.get(h, 0)
        if count == 0 or self.total == 0:
            return 1
        # Simple: log-frequency mapped to 1-256
        ratio = math.log(count + 1) / math.log(max(self.counts.values()) + 1)
        return max(1, min(num_classes, int(ratio * num_classes) + 1))


# ============================================================
# FINGERPRINT PARSER
# ============================================================

class FingerprintParser:
    """
    Converts token strings → StructuralFingerprint tuples.
    
    No embedding matrix. No learned parameters.
    Just hashing + classification + counting.
    """
    
    def __init__(self):
        self.freq_tracker = FrequencyTracker()
    
    def fit(self, corpus_tokens: List[str]):
        """Learn frequency distribution from a corpus."""
        self.freq_tracker.observe_batch(corpus_tokens)
        print(f"  Frequency tracker: {len(self.freq_tracker.counts)} unique tokens seen")
    
    def parse(self, token: str, pos: int) -> StructuralFingerprint:
        """Convert a single token to its structural fingerprint."""
        h = djb2_hash(token)
        fc = self.freq_tracker.freq_class(token)
        tc = type_of_token(token)
        return StructuralFingerprint(hash=h, freq_class=fc, type_class=tc, pos=pos)
    
    def parse_sequence(self, tokens: List[str]) -> List[StructuralFingerprint]:
        """Convert a token sequence to fingerprints."""
        return [self.parse(t, i) for i, t in enumerate(tokens)]
    
    def fingerprints_to_tuples(self, fps: List[StructuralFingerprint]) -> List[Tuple[int, int, int, int]]:
        """Convert fingerprints to plain tuples for fast processing."""
        return [fp.to_tuple() for fp in fps]


# ============================================================
# DEMO / TEST
# ============================================================

def demo():
    """Simple demo showing the parser works with zero Transformer components."""
    print("=" * 60)
    print("Nova V2 — Phase 1: Structural Fingerprint")
    print("=" * 60)
    print()
    print("CONSTRAINTS:")
    print("  ❌ No embedding matrix")
    print("  ❌ No d_model / hidden_size")
    print("  ❌ No vector lookup")
    print("  ✅ Only: hash + freq_class + type_class + pos")
    print()
    
    # Sample tokens
    tokens = ["The", "cat", "sat", "on", "the", "mat", ".", "42", "hello"]
    
    # Learn frequencies
    parser = FingerprintParser()
    parser.fit(tokens)
    
    # Parse each token
    print("Token → Fingerprint:")
    for i, t in enumerate(tokens):
        fp = parser.parse(t, i)
        print(f"  '{t:8s}' → {fp}")
    
    print()
    print("Structural distance between 'cat' and 'sat':")
    fp1 = parser.parse("cat", 0)
    fp2 = parser.parse("sat", 0)
    print(f"  distance(cat, sat) = {fp1.distance(fp2)}")
    
    print()
    print("Same word 'the' at different positions:")
    fp3 = parser.parse("the", 2)
    fp4 = parser.parse("the", 5)
    print(f"  'the' at pos 2: {fp3}")
    print(f"  'the' at pos 5: {fp4}")
    print(f"  distance = {fp3.distance(fp4)} (only position differs)")
    
    print()
    fp_size = 4 * 4  # 4 ints × 4 bytes
    embed_size = 2048 * 4  # typical embedding: 2048 floats × 4 bytes
    print(f"Memory per token: {fp_size} bytes (vs {embed_size} bytes for embedding)")
    print(f"Savings: {embed_size // fp_size}x less memory")
    
    return parser


if __name__ == '__main__':
    demo()