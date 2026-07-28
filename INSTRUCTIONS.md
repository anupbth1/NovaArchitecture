# NovaArchitecture - Complete Usage Guide

## Two Architectures in This Repo

| Version | Status | Description |
|---------|--------|-------------|
| **RCV v1** | ⚠️ Frozen research baseline | Transformer variant. Corrective iteration LM. Working but slow. |
| **Nova V2** | 🚧 In development (Phases 1-3 done) | **Zero Transformer components.** Structural computation. |

---

## V2: Zero Transformer Components (New Architecture)

### Concept

```diff
- No embedding matrix (16 bytes/token vs 8192 bytes)
- No hidden state vectors (set of bindings vs 2048 floats)
- No Attention / MLP (pattern matching vs matrix multiply)
- No backpropagation (rule weight updates vs gradient descent)
+ Only: integers, sets, patterns, and rules
```

---

### 🟢 Colab Setup (Free T4 GPU) - 2 Minutes

Open https://colab.research.google.com/ → New Notebook → Runtime → Change runtime type → **T4 GPU**

```python
# ───────────────────────────────────────────
# CELL 1: Clone and setup
# ───────────────────────────────────────────
!git clone https://github.com/anupbth1/NovaArchitecture.git
%cd NovaArchitecture

# ───────────────────────────────────────────
# CELL 2: Run V2 demo (zero Transformer components)
# ───────────────────────────────────────────
!python nova/v2/demo_all.py

# Output:
#   PARSING TOKENS -> FINGERPRINTS (16 bytes/token vs 8192)
#   WORKING MEMORY (set of bindings vs hidden states)
#   RULE ENGINE (pattern matching vs attention/MLP)
#   Zero Transformer components confirmed

# ───────────────────────────────────────────
# CELL 3: Run V2 Phase 1 (Fingerprint Parser)
# ───────────────────────────────────────────
!python nova/v2/fingerprint.py

# ───────────────────────────────────────────
# CELL 4: Run V2 Phase 2 (Working Memory)
# ───────────────────────────────────────────
!python nova/v2/memory.py

# ───────────────────────────────────────────
# CELL 5: Run V2 Phase 3 (Rule Engine)
# ───────────────────────────────────────────
!python nova/v2/rules.py

# ───────────────────────────────────────────
# CELL 6: Run V1 tests (optional - V1 is frozen)
# ───────────────────────────────────────────
!python tests/test_rcv.py

# ───────────────────────────────────────────
# CELL 7: V1 CLI (V1 is frozen, but works)
# ───────────────────────────────────────────
!python -m nova.cli train --model tiny --steps 200 --device cuda
```

---

### 🟢 RunPod Setup (A100 - for serious training)

```bash
# Start: RunPod → GPU Pod → A100 (40GB) → PyTorch template

# Clone and setup
git clone https://github.com/anupbth1/NovaArchitecture.git
cd NovaArchitecture

# Run V2 demo
python nova/v2/demo_all.py

# Run individual V2 phases
python nova/v2/fingerprint.py
python nova/v2/memory.py
python nova/v2/rules.py

# V1 training (frozen, optional)
pip install torch numpy
python -m nova.cli train --model small --data tiny_stories --steps 5000 --device cuda
```

---

### 🟢 Local PC Setup

```bash
# Prerequisites: Python 3.10+, pip, git

# Clone
git clone https://github.com/anupbth1/NovaArchitecture.git
cd NovaArchitecture

# V2 runs without any dependencies (pure Python - no PyTorch needed for V2!)
python nova/v2/demo_all.py

# Optional: install PyTorch for V1
pip install torch numpy
python -m nova.cli train --model tiny --steps 200
```

---

## V2 Phase-by-Phase Explanation

### Phase 1: Fingerprint Parser (`nova/v2/fingerprint.py`)

**What it does**: Converts tokens to structural fingerprints without embeddings.

```python
from nova.v2.fingerprint import FingerprintParser

parser = FingerprintParser()
parser.fit(["The", "cat", "sat"])  # Learn frequencies

fp = parser.parse("cat", pos=0)
print(fp.hash, fp.freq_class, fp.type_class, fp.pos)
# Output: 550141, 255, 0, 0
#          ^hash  ^freq    ^type ^pos
```

**No embedding matrix.** Just hashing + counting + classification.

---

### Phase 2: Working Memory (`nova/v2/memory.py`)

**What it does**: Maintains state as a set of (role, value) bindings.

```python
from nova.v2.memory import WorkingMemory, RoleVocabulary

roles = RoleVocabulary()
memory = WorkingMemory()
memory.add(roles.get_id("SUBJECT"), 550141)  # SUBJECT = cat
memory.add(roles.get_id("VERB"), 567565)      # VERB = sat

matches = memory.match(role=roles.get_id("SUBJECT"))
print(matches)  # [(10, 550141)]
```

**No hidden state vectors.** Just a dynamic set of bindings.

---

### Phase 3: Rule Engine (`nova/v2/rules.py`)

**What it does**: Pattern matching replaces attention + MLP.

```python
from nova.v2.rules import RuleEngine, Pattern, Action, Rule
from nova.v2.memory import RoleVocabulary

roles = RoleVocabulary()
engine = RuleEngine()

# Rule: if position < 5, mark as "early"
rule = Rule(
    pattern=Pattern(value_constraints=[(roles.get_id("POSITION"), 'lt', 5)]),
    action=Action(add=[(roles.get_id("CONTEXT"), 1)]),
    weight=0.5,
)
engine.add_rule(rule)

memory, trace = engine.forward_with_trace(memory)
```

**No matrix multiplication.** No QKV. No GELU. Just integer comparisons.

---

## API Reference

### V2 Components

| Class | File | Purpose |
|-------|------|---------|
| `FingerprintParser` | `fingerprint.py` | Token → (hash, freq, type, pos) |
| `StructuralFingerprint` | `fingerprint.py` | 4-integer token representation |
| `FrequencyTracker` | `fingerprint.py` | Learn token frequency distribution |
| `WorkingMemory` | `memory.py` | Dynamic set of (role, value) bindings |
| `RoleVocabulary` | `memory.py` | Role name ↔ integer ID mapping |
| `Pattern` | `rules.py` | Condition on memory state |
| `Action` | `rules.py` | Transformation on memory |
| `Rule` | `rules.py` | Pattern → Action + learned weight |
| `RuleEngine` | `rules.py` | Applies all rules to memory |

### V1 Components (Frozen)

| Command | Purpose |
|---------|---------|
| `python -m nova.cli train --model tiny` | Train tiny V1 model |
| `python -m nova.cli test` | Run V1 tests |
| `python -m nova.cli convert --from gpt2` | Convert GPT-2 to V1 |

---

## File Structure

```
nova/v2/              # V2: Zero Transformer components (Phases 1-3 done)
  __init__.py
  fingerprint.py      # Phase 1: Token → 4 integers (16 bytes vs 8192)
  memory.py           # Phase 2: Set of (role, value) bindings
  rules.py            # Phase 3: Pattern matching rule engine
  demo_all.py         # End-to-end demo of Phases 1-3

nova/rcv/             # V1: Frozen research baseline
  brain_cell.py       # Transformer-like MLP
  slot_debate.py      # Transformer-like attention
  reasoner.py         # Corrective iteration loop
  nova_brain.py       # Complete V1 model
  config.py           # Configuration
  trainer.py          # Training pipeline

nova/cli.py           # CLI: python -m nova.cli [command]
```

---

## Requirements

| Component | Dependencies |
|-----------|-------------|
| **V2 (Phases 1-3)** | **None! Pure Python** (no PyTorch needed) |
| V1 training | `pip install torch numpy` |
| V1 CLI | `pip install torch numpy` |
| Benchmarks | `pip install torch numpy` |
| Convert existing | `pip install transformers` |

---

## Known Issues

| Issue | Affects | Status |
|-------|---------|--------|
| Training not converging | V1 only | Frozen - documented in audit |
| Slow sequential processing | V1 only | Architectural limitation |
| Zero Transformer rule | V2 | ✅ Phases 1-3 pass |
| Learning algorithm (Phase 4) | V2 | Not yet implemented |
| Text generation (Phase 5) | V2 | Not yet implemented |

---

## Quick Reference Card

### Colab
```python
!git clone https://github.com/anupbth1/NovaArchitecture.git
%cd NovaArchitecture
!python nova/v2/demo_all.py                          # V2 demo
!python -m nova.cli test                              # V1 tests
!python -m nova.cli train --model tiny --steps 200    # V1 train
```

### Local
```bash
git clone https://github.com/anupbth1/NovaArchitecture.git
cd NovaArchitecture
python nova/v2/demo_all.py            # V2 (no install needed)
python -m nova.cli test                # V1 (needs torch)
```

### RunPod
```bash
git clone https://github.com/anupbth1/NovaArchitecture.git
cd NovaArchitecture
python nova/v2/demo_all.py
python -m nova.cli train --model small --data tiny_stories --steps 5000 --device cuda