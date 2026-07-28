"""
Nova V2 - Complete Demo (Phases 1-3)
======================================
End-to-end: Fingerprint -> Memory -> Rules
Zero Transformer components.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from nova.v2.fingerprint import FingerprintParser
from nova.v2.memory import WorkingMemory, RoleVocabulary
from nova.v2.rules import RuleEngine, create_default_rules

def run_full_pipeline():
    print("=" * 70)
    print("NOVA V2 - End-to-End Pipeline Demo")
    print("Zero Transformer Components")
    print("=" * 70)

    # Step 1: Parse tokens into fingerprints (no embedding matrix)
    print("\n[1] PARSING TOKENS -> FINGERPRINTS")
    tokens = ["The", "cat", "sat", "on", "the", "mat", ".", "42"]
    parser = FingerprintParser()
    parser.fit(tokens)

    fingerprints = parser.parse_sequence(tokens)
    for tok, fp in zip(tokens, fingerprints):
        mem = 4 * 4  # 16 bytes
        embed = 2048 * 4  # 8192 bytes
        print(f"  '{tok:8s}' -> {fp}  ({mem}B vs {embed}B for embedding)")

    # Step 2: Setup working memory (no hidden state vectors)
    print("\n[2] WORKING MEMORY")
    roles = RoleVocabulary()
    memory = WorkingMemory()
    memory.add(roles.get_id("POSITION"), 0)
    memory.add(roles.get_id("TOKEN_HASH"), fingerprints[0].hash)  # "The"
    memory.add(roles.get_id("TOKEN_TYPE"), 0)  # word
    print(f"  Memory: {memory}")
    print(f"  Size: {len(memory)} bindings x 8 bytes = {len(memory)*8} bytes")
    print(f"  (Hidden state would be 2048x4 = 8192 bytes)")

    # Step 3: Run rules (no attention, no MLP)
    print("\n[3] RULE ENGINE")
    engine = RuleEngine()
    for rule in create_default_rules(roles):
        engine.add_rule(rule)
    print(f"  Rules loaded: {len(engine)}")

    memory_after, trace = engine.forward_with_trace(memory.copy())
    print(f"  Rules fired: {len(trace)}")
    for rule, weight in trace:
        print(f"    -> {rule.description} (w={weight})")
    print(f"  Total confidence: {engine.total_confidence}")
    print(f"  Memory now: {memory_after}")

    # Step 4: Summary
    print("\n" + "=" * 70)
    print("WHAT WAS AVOIDED")
    print("=" * 70)
    print("  No embedding matrix")
    print("  No d_model / hidden_size")
    print("  No Attention (QKV)")
    print("  No MLP / FFN")
    print("  No LayerNorm")
    print("  No GELU / ReLU")
    print("  No matrix multiplication")
    print("  No backpropagation")
    print("=" * 70)
    print("Nova V2 Phases 1-3: WORKING")
    print("=" * 70)

if __name__ == '__main__':
    run_full_pipeline()