# NovaArchitecture Mathematics

## Core RCV Equation

The fundamental recurrence that powers NovaArchitecture:

```
workspace_{t+1} = BrainCell(Debate(workspace_t)) + workspace_t
```

Where:
- `workspace_t ∈ ℝ^{B × M × D}` — 64 memory slots at iteration t
- `BrainCell: ℝ^{B × M × D} → ℝ^{B × M × D}` — Dense MLP applied per slot
- `Debate: ℝ^{B × M × D} → ℝ^{B × M × D}` — Self-attention among slots

## Effective Depth

Traditional Transformers stack L layers:
```
Effective Depth = L  (e.g., L=96 for GPT-3)
```

Nova RCV reuses a single layer N times:
```
Effective Depth = N  (e.g., N=50 for full model)
```

## Parameter Count vs Effective Compute

**Training Parameters (P):**
```
P = P_embed + P_brain + P_debate + P_mixer + P_head

P_embed = vocab_size × d_model
P_brain = 2 × d_model² × expansion + 2 × d_model
P_debate = 4 × d_model²  
P_mixer = 2 × d_model²
P_head = d_model × vocab_size
```

**Effective Compute per Token (C_eff):**
```
C_eff = P × N × M

Where:
- P = total parameters
- N = iterations per token
- M = memory slots (64)

At scale (d_model=2048, expansion=8, N=50, M=64):
P ≈ 800M
C_eff = 800M × 50 × 64 = 2.56T FLOPs equivalent
```

## Deep Supervision Loss

```
L_total = α · L_final + (1-α) · L_deep

L_final = CE(logits_N, target)  # Final iteration only
L_deep = (1/N) · Σ_i w_i · CE(logits_i, target)  # All iterations

Where:
- α = 0.6 (final loss weight)
- w_i = linearly increasing from 0.1 to 1.0
- CE = cross-entropy loss
- logits_i = Head(workspace_i[0])  # First slot projection
```

## Adaptive Compute (Surprisal-Based)

```
iter_limit = f(surprisal)

surprisal_t = (1-β) · surprisal_{t-1} + β · L_t

iter_limit = min_iter + (max_iter_cap - min_iter) · σ(difficulty)

Where:
- β = 0.1 (EMA decay)
- σ = sigmoid function
- difficulty = ComputePredictor(avg_workspace_state)
```

## Slot Debate Mechanism

Standard self-attention among M=64 slots:

```
Q, K, V = Split(QKV(workspace))  # Q, K, V ∈ ℝ^{B × M × D}
attn = softmax(QK^T / √D)  # ℝ^{B × M × M}
out = attn · V  # ℝ^{B × M × D}
workspace_new = workspace + Proj(out)  # Residual connection
```

Complexity: O(M² · D) = O(64² · 2048) = 8.4M FLOPs
(Compare to standard attention: O(T² · D) = O(256² · 2048) = 134M FLOPs)

## Position Encoding

Learned position embeddings rather than sinusoidal:

```
x_t = Embed(token_t) · √d_model + PosEmbed(t)
```

This is simpler and works well for sequential (non-parallel) processing.

## Initialization

All weights initialized with small values for iterative stability:

```
W ∼ N(0, 0.01)  # Linear layers
b = 0            # Biases
γ = 1, β = 0     # LayerNorm
```

## Gradient Flow

Gradient clipping at 1.0 norm is critical for iterative models:
```
if ||g|| > 1.0: g = g · (1.0 / ||g||)
```

This prevents gradient explosion over the unrolled iterations.