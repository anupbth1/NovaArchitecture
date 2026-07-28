# Nova V2 — Design Philosophy

## Status: 🔜 Design Phase — No Code

This document defines the **mathematical specification** for Nova V2.

**No code will be written until this specification is complete.**

---

## Core Constraint

```
ZERO Transformer Components:
❌ Attention (QKV, multi-head, linear attention, any quadratic mechanism)
❌ MLP / FFN blocks (dense → expansion → dense)
❌ LayerNorm / RMSNorm / BatchNorm
❌ GELU / ReLU / SiLU / SwiGLU / any standard activation
❌ Residual connections (skip connections, add & norm)
❌ Positional encoding (learned, sinusoidal, RoPE, ALiBi)
❌ KV cache (key-value storage for past tokens)
❌ Learned embeddings as primary representation
❌ Autoregressive next-token prediction as the only training objective
❌ Cross-entropy loss on token logits
❌ Backpropagation through time (BPTT) for training
❌ Gradient descent on static computational graphs
```

---

## The Problem We Are Solving

### What's wrong with Transformers?

1. **Quadratic attention** — O(T²·D) is the fundamental bottleneck. Everything else is a patch.
2. **Token-level representation** — "The" and "The" have the same vector regardless of context. Meaning is lost.
3. **Static computational graph** — Every token follows the same path. No dynamic computation.
4. **Backpropagation** — Requires storing the entire computation graph. Memory O(T·L·D).
5. **Autoregressive generation** — O(T) sequential steps cannot be parallelized.
6. **Knowledge in weights** — All knowledge is baked into parameters during training. Cannot update at inference.
7. **No causal understanding** — Correlations, not causation. Pattern matching, not reasoning.

### What Nova V2 must solve differently

| Problem | Transformer approach | V2 must find |
|---------|---------------------|--------------|
| Sequence processing | Parallel with attention | ? |
| Memory | KV cache of all tokens | ? |
| Representation | Token → Vector (fixed) | ? |
| Learning | Backprop through graph | ? |
| Generation | Autoregressive | ? |
| Knowledge | Frozen in weights | ? |
| Reasoning | Emergent from scale | ? |

---

## Design Requirements

Every module in V2 must answer:

1. **Why it exists** — What fundamental problem does it solve?
2. **What Transformer problem it fixes** — How is this different from what Transformers do?
3. **Time complexity** — O(?) in sequence length, dimension, vocabulary
4. **Memory complexity** — O(?) at training and inference
5. **Training algorithm** — How does it learn? (Cannot be BPTT)
6. **Inference algorithm** — How does it generate? (Cannot be autoregressive)
7. **Scale argument** — Why should this work better with more compute/data?

---

## Proposed Architecture Direction

### Core Idea: Semantic Atoms over Token Vectors

Instead of:
```
Token → Embedding Vector → Attention → MLP → Vector → Softmax → Token
```

Think:
```
Token → Semantic Atom → Dynamic State → Reasoning → Action → Output
```

### Key Differences

| Concept | Transformer | Nova V2 |
|---------|------------|---------|
| Representation | Fixed-size vector per token | Variable-binding semantic atom |
| State | Hidden states from previous layer | World model (situational, not positional) |
| Computation | Same for all tokens | Dynamic per atom |
| Memory | KV cache of tokens | Structural memory of concepts |
| Learning | Gradient on graph | Rule extraction + consolidation |
| Generation | Next token | Goal-directed action sequence |
| Knowledge | Parameter weights | Dynamic memory + learned operators |

### What "Semantic Atom" means

A semantic atom is NOT a vector embedding. It is:
- A **binding** between a symbol and a role
- Compositional: `(IS-A, "cat")`, `(LOCATION, "mat")`, `(COLOR, "orange")`
- Dynamically created and destroyed during computation
- Not indexed by position but by meaning
- Can be compared, merged, split, and transformed

### What "Dynamic State" means

Instead of a fixed-size hidden state `h_t ∈ ℝ^D`, the state is:
- A **dynamic collection** of semantic atoms (like working memory)
- Size changes per timestep (no fixed bottleneck)
- Operations on atoms are what Transformers do with vectors
- No matrix multiplication — discrete operations guided by learned rules

### Computation Model (preliminary)

```
1. Input token → Parse into semantic atoms
2. Atoms enter working memory
3. Reasoning rules fire based on atom patterns
4. Rules create, modify, or delete atoms
5. When a "respond" rule fires, atoms → output
6. Learning = discovering which rules to fire when
```

This is closer to a **neural-symbolic system** than a neural network.

---

## Phases

### Phase 0: Mathematical Specification
- Define the atom algebra
- Define the reasoning rule language
- Define the learning objective
- Prove complexity bounds
- [No code]

### Phase 1: Semantic Representation
- Implement the atom parsing system
- Must handle: text, code, math, vision concepts
- Output: structured atoms, not vectors
- [No attention, no embeddings]

### Phase 2: Dynamic Memory
- Working memory with atom operations
- Long-term memory with consolidation
- Retrieval based on structural matching, not vector similarity
- [No KV cache, no attention]

### Phase 3: Reasoning Runtime
- Rule engine that operates on atoms
- Forward chaining, backward chaining, conflict resolution
- Learned rule weights (not gradient descent)
- [No MLP, no activation functions]

### Phase 4: Learning Algorithm
- How rules are discovered
- How rule weights are updated
- Not backpropagation
- Must work online (continual learning)
- [No BPTT, no gradient descent]

### Phase 5: Language Generator
- Convert atoms back to language
- Goal-directed generation (not autoregressive)
- Can generate different outputs from same state
- [No softmax over vocabulary]

### Phase 6: Multimodal
- Vision, audio as input → same atom representation
- Born multimodal, not adapted
- [No separate vision encoder]

---

## Rule: No Code Until Math is Complete

Every module must have a written specification covering:

```
1. Mathematical model
2. Data structures
3. Time complexity
4. Memory complexity
5. Learning algorithm
6. Inference algorithm
7. Proof of scaling
```

Only after ALL Phases 0-6 have specifications does implementation begin.

---

## Timeline (Estimated)

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 0: Math Spec | 2 weeks | Complete mathematical model |
| 1: Semantic Rep | 1 week | Atom parser working |
| 2: Dynamic Memory | 2 weeks | Working memory system |
| 3: Reasoning | 3 weeks | Rule engine functional |
| 4: Learning | 4 weeks | Learning algorithm converges |
| 5: Generation | 2 weeks | Text output from atoms |
| 6: Multimodal | 4 weeks | Image input working |

**Total**: ~18 weeks for first working prototype

---

## Why This Will Be Harder Than V1

V1 was hard because gradient flow through iterations failed.
V2 will be hard because we are **rejecting the entire neural network paradigm** that has dominated AI for a decade.

| Challenge | Risk | Mitigation |
|-----------|------|------------|
| No gradient descent | Very high — we don't know what replaces it | Start with simple rule learning |
| No attention | Very high — attention is the best mechanism we have | Use structural matching instead |
| No embeddings | High — how to represent words? | Symbolic groundings with learned mappings |
| Performance | High — discrete operations are slower on GPU | Hybrid approach if needed |
| Community skepticism | Medium — everyone uses Transformers | Publish results honestly |

---

## Final Note

If Nova V2 uses even ONE Transformer component, we will admit it is a Transformer variant and stop calling it novel.

**The goal is not to "beat Transformers."**

**The goal is to find a DIFFERENT computation model.**

If it doesn't work, that's research. If it works, that's a breakthrough.

Both outcomes are valuable.