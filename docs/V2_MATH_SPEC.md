# Nova V2 — Mathematical Specification

**Status**: 🔜 Phase 0 — Mathematical specification in progress  
**Rule**: Zero Transformer components. Zero code until spec is complete.  
**Goal**: 1B-equivalent FLOPs, 600B-equivalent accuracy, low-end GPU.

---

## 1. Core Philosophy

### 1.1 The Problem with Transformers

Transformers compute via:
```
output = softmax(QK^T / √d) · V · W
```
This is **continuous linear algebra on fixed-size vectors**. It requires:
- Large matrix multiplications (O(d²))
- Fixed-size hidden states (d_model)
- Gradient backpropagation through the entire graph
- Massive parameter counts for knowledge storage

### 1.2 Nova V2 Approach

Transformers store knowledge in **weights**. Nova V2 stores knowledge in **structure**.

Instead of:
```
Token → Vector → Matrix multiply → Vector → Token
```

We do:
```
Token → Structural Atom → Pattern Match → Transform → Token
```

No vectors. No matrices. No gradients. Just **structural operations on discrete representations**.

---

## 2. The Computation Model

### 2.1 Input Representation

**Problem**: How to represent a token without an embedding matrix?

**Solution**: Each token is represented as a **structural fingerprint** — a tuple of primitive features:

```
token_t = (pos, hash, freq_class, type_class)
```

Where:
- `pos` = position in sequence (integer)
- `hash` = hash of token string (e.g., DJB2 hash mod 2²⁰)
- `freq_class` = frequency quantile (1–256, estimated from corpus)
- `type_class` = token type (0=word, 1=number, 2=punctuation, 3=special)

**No embedding matrix.** No vector lookup. Just 4 integers per token.

**Complexity**: O(1) per token — just hashing, no matrix multiply.

### 2.2 Working Memory

**Problem**: How to maintain state without hidden vectors?

**Solution**: Working memory is a **set of typed bindings** — not a fixed-size vector.

```
M = {(role₁, value₁), (role₂, value₂), ..., (role_k, value_k)}
```

Where:
- `role` = a semantic role (integer ID from a learned role vocabulary)
- `value` = a structural atom (integer or tuple of integers)
- The set has **variable size** — not fixed to d_model

**Operations on M**:
- `add(M, (role, value))` — insert binding
- `remove(M, role)` — delete binding
- `match(M, pattern)` — find bindings matching a pattern
- `merge(M₁, M₂)` — union of two memory sets

**No hidden state vector.** Just a growing/shrinking set.

**Complexity**: O(|M|) for linear scan, O(log|M|) with indexing.

### 2.3 Computation Unit

**Problem**: What replaces the MLP/Attention?

**Solution**: **Rule application via structural matching.**

The computation is a set of rules:
```
R = {r₁, r₂, ..., r_n}
```

Each rule has:
- **Pattern**: a condition on the current memory M
- **Action**: transform M by adding/removing bindings
- **Weight**: learned strength of the rule (scalar, not a matrix)

**Forward pass** (given input token → memory M):

```
for each rule r in R:
    if r.pattern matches M:
        apply r.action to M
        accumulate r.weight to confidence
```

**This replaces both attention AND MLP with a single operation: structural pattern matching.**

**No matrix multiplication.** Just set operations and integer comparisons.

**Complexity**: O(|R| · |M|) — number of rules × memory size.  
For |R| = 10K, |M| = 1K: ~10M comparisons — **fits on any GPU**.

### 2.4 What "Pattern Matching" Means

Pattern = tuple of constraints:
```
pattern = (role, value_constraint, context_constraints)
```

Example pattern:
```
pattern = (ROLE_SUBJECT, ANY, {
    (ROLE_VERB, "runs"),
    (ROLE_TENSE, "present")
})
```

This matches "any subject where the verb is 'runs' in present tense."

**No dot products. No cosine similarity. Just exact/structural matching.**

### 2.5 Learning Algorithm

**Problem**: How to learn without backpropagation?

**Solution**: **Three-stage consolidation:**

1. **Observation** (during forward pass):
   - When a rule fires, record: (pattern, action, outcome)
   - Outcome = reward (+1 if correct prediction, -1 if wrong, 0 otherwise)

2. **Rule Weight Update** (after each batch):
   ```
   w_i = w_i + α · (reward_i - base_rate) · firing_count_i
   ```
   Where:
   - `w_i` = weight of rule i (scalar, starting at 0)
   - `α` = learning rate (single scalar, not a tensor)
   - `reward_i` = average reward when rule i fires
   - `base_rate` = expected reward from random
   - `firing_count_i` = how often rule i fires

   **This is NOT gradient descent.** It's a simple online learning rule — like a bandit algorithm.

3. **Rule Discovery** (periodic):
   - Find patterns that frequently co-occur but have no rule
   - Create new rules for them
   - Prune rules with weight < threshold

**No backpropagation. No computational graph. No gradient.**

**Complexity**: O(|R|) per update — linear in number of rules.

### 2.6 Output Generation

**Problem**: How to generate tokens without softmax over vocabulary?

**Solution**: **Constrained structural generation.**

After processing all tokens through the rule system, the memory M contains bindings representing the predicted next token's properties:

```
M = {
    (PREDICTED_HASH, h),
    (PREDICTED_POS, p),
    (CONFIDENCE, c),
    ...
}
```

To generate the actual token:
1. Look up `PREDICTED_HASH` in a **reverse index** (hash → token)
2. If multiple tokens match the hash, use other bindings (`TYPE_CLASS`, `FREQ_CLASS`) to disambiguate
3. If confidence < threshold, generate multiple candidates and beam search

**The reverse index is NOT an embedding matrix.** It's a hash table:
```
hash_table[hash] = [(token₁, freq₁), (token₂, freq₂), ...]
```

**No softmax over 50K vocabulary.** Just hash lookup → O(1).

### 2.7 Positional Information

**Problem**: How to handle position without positional encoding?

**Solution**: Position is just another binding in the working memory:
```
M = M ∪ {(POSITION, t)}
```

Where `t` is the current token index (integer). Rules can match on position:
```
pattern = (ROLE_VERB, ANY, {(POSITION_RANGE, (5, 10))})
```

**No sinusoidal encoding. No learned position embeddings.** Just an integer in a set.

---

## 3. Complexity Analysis

### 3.1 Time Complexity

| Operation | Transformer (d=2048, L=32) | Nova V2 (|R|=10K, |M|=1K) |
|-----------|---------------------------|---------------------------|
| Input encoding | O(vocab × d) = 100M | O(1) hash + 4 ints |
| Per-token compute | O(L × d²) = 134M FLOPs | O(|R| × |M|) = 10M comparisons |
| Attention | O(T × d) = 0.5M FLOPs | Not needed |
| Learning | O(d²) backprop = 4M FLOPs | O(|R|) weight update = 10K ops |
| Output | O(vocab × d) = 100M | O(1) hash lookup |
| **Total per token** | **~240M FLOPs** | **~10M comparisons** |

### 3.2 Memory Complexity

| Aspect | Transformer (1B params) | Nova V2 |
|--------|------------------------|---------|
| Parameters | 1B floats = 4 GB | |R| scalars = 40 KB + rules = ~100 MB |
| KV cache | O(T × d) = 2K × 2048 = 16 MB | None needed |
| Working memory | Hidden states = 8 MB | |M| bindings = ~100 KB |
| **Total** | **~4 GB** | **~100 MB** |

### 3.3 FLOPs Comparison

| Model | FLOPs per token | Hardware |
|-------|----------------|----------|
| 600B Transformer | ~1.2 TFLOPs | 8× A100 cluster |
| Qwen2.5-1.5B | ~3 GFLOPs | 1× A100 |
| **Nova V2** | **~10 M comparisons** | **1× T4 (15 GB)** |

---

## 4. Accuracy Argument

### 4.1 Why Structural Computation Can Beat Linear Algebra

Transformers have a fundamental limitation: **everything is a vector**. "The" and "The" in different contexts have different vectors, but they're stored in the same matrix. This leads to:

- Context mixing: all knowledge is blended into weights
- Interference: learning one thing can overwrite another
- No explicit reasoning: patterns are statistical, not structural

Nova V2 has:
- **Explicit rules**: Each rule represents a discrete piece of knowledge
- **No interference**: Rules don't blend; they compete by weight
- **Compositional**: Rules can chain (rule A creates bindings that rule B matches)
- **Interpretable**: You can read the rules and understand what the model knows

### 4.2 Scaling Hypothesis

Transformers scale because more parameters = more knowledge storage.  
Nova V2 scales because more rules = more knowledge patterns.

Key difference:
- **Transformer scaling**: O(N²) compute for N parameters (matrix multiply)
- **Nova V2 scaling**: O(N) compute for N rules (linear scan)

So Nova V2 can have **100× more knowledge patterns than a Transformer has parameters** for the same compute budget.

### 4.3 FLOP Efficiency Hypothesis

A 600B Transformer uses 1.2 TFLOPs per token. Most of this is **wasted on irrelevant computations** — every token attends to every other token, every layer processes every dimension.

Nova V2 only computes **relevant rules** triggered by the current memory state. If a rule's pattern doesn't match, it's **skipped entirely** — no computation wasted.

**Estimated efficiency**: Nova V2 should achieve comparable accuracy to a model that is **100–1000× larger in parameter count**, because:
1. No computation is wasted on irrelevant dimensions
2. Knowledge is stored discretely, not blended into matrices
3. Rules can be chained compositionally, creating exponential reasoning depth

---

## 5. Architecture Summary

```
Input Token
    │
    ▼
Structural Fingerprint (pos, hash, freq, type)   ← No embedding matrix
    │
    ▼
Working Memory M (set of bindings)                ← No hidden state vector
    │
    ▼
Rule Application (pattern matching)               ← No attention, no MLP
    │
    ▼
Memory Update (add/remove bindings)               ← No residual connections
    │
    ▼
Output Generation (hash lookup)                    ← No softmax head
    │
    ▼
Output Token
```

**Training** (no backpropagation):
1. Forward pass → record which rules fired
2. Compare output to target → compute reward
3. Update rule weights: w_i += α · (reward - base) · count_i
4. Periodically discover new rules, prune weak ones

---

## 6. Open Questions (To Be Resolved in V2 Development)

| Question | Status |
|----------|--------|
| Rule vocabulary size? (How many roles?) | TBD in Phase 1 |
| Rule discovery algorithm? (How to find patterns?) | TBD in Phase 4 |
| Reward function? (Beyond next-token prediction) | TBD in Phase 4 |
| Beam search for generation? (How many candidates?) | TBD in Phase 5 |
| Multimodal input? (Images/audio as structural atoms) | TBD in Phase 6 |
| Hardware optimization? (GPU-friendly set operations) | TBD in Phase 3 |

---

## 7. Timeline

| Phase | Duration | Milestone |
|-------|----------|-----------|
| **0: Math Spec** | ✅ Complete | This document |
| 1: Structural Rep | 1 week | Fingerprint parser working |
| 2: Working Memory | 2 weeks | Set operations + indexing |
| 3: Rule Engine | 3 weeks | Pattern matching + rule application |
| 4: Learning | 4 weeks | Weight updates + rule discovery |
| 5: Generation | 2 weeks | Hash lookup + beam search |
| 6: Multimodal | 4 weeks | Image/audio → structural atoms |

---

*"No vectors. No matrices. No gradients. Just patterns and rules."*