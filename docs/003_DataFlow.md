# NovaArchitecture Data Flow

## End-to-End Processing Pipeline

```
┌─────────────┐
│   Input     │  Raw text or tokens
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Tokenizer  │  GPT-2 compatible tokenization
│  (Optional) │  (50257 vocab)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Embedding  │  Token → Vector (B, T) → (B, T, D)
│  + PosEnc   │  Scaled embedding (×√D) + learned position
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Workspace │  Initialize 64 memory slots
│   Init      │  (B, 64, D) learned parameters
└──────┬──────┘
       │
       ▼
  ┌────┴────┐    ┌─────────────────────────────┐
  │ Token   │    │  For each token t in sequence:│
  │ Loop    │───▶│                              │
  └─────────┘    │  1. Inject token into slots  │
                 │  2. Loop N iterations:        │
                 │     a. BrainCell (knowledge)  │
                 │     b. SlotDebate (perspect.) │
                 │  3. Read first slot → predict │
                 │  4. Update workspace          │
                 └──────────────────────────────┘
                                    │
                                    ▼
                            ┌───────────────┐
                            │   Output Head  │  LayerNorm + Linear
                            │   (B, T, V)    │  → token logits
                            └───────┬───────┘
                                    │
                                    ▼
                            ┌───────────────┐
                            │   Sampling    │  Top-k, Top-p, Temperature
                            │   or Loss     │  Cross-entropy or argmax
                            └───────┬───────┘
                                    │
                                    ▼
                            ┌───────────────┐
                            │   Output      │  Generated text or loss
                            └───────────────┘
```

## Data Shapes

| Stage | Shape | Description |
|-------|-------|-------------|
| Input tokens | `(B, T)` | Batch of token sequences |
| Embedding | `(B, T, D)` | Token vectors + position encoding |
| Workspace | `(B, M, D)` | M=64 memory slots, D=d_model |
| Token injection | `(B, M, D)` | Token broadcast to all slots |
| BrainCell | `(B, M, D)` → `(B, M, D)` | MLP applied per slot |
| SlotDebate | `(B, M, D)` → `(B, M, D)` | Self-attention among slots |
| Iter outputs | `(B, N, D)` | N iterations of first-slot states |
| Logits | `(B, N, V)` | N iterations projected to vocabulary |
| Final logits | `(B, T, V)` | Concatenated across tokens |

## Key Innovation: Per-Token Iterative Processing

Unlike Transformers that process all tokens in parallel with O(T²·D) attention,
RCV processes tokens **sequentially** with O(T·N·M²·D) compute.

- T = sequence length (256-2048)
- N = iterations per token (5-50, adaptive)
- M = memory slots (64)
- D = hidden dimension (2048)

**Traditional Transformer**: O(T²·D) = 256² × 2048 ≈ 134M FLOPs/token
**Nova RCV**: O(T·N·M²·D) = 256 × 30 × 64² × 2048 ≈ 64B FLOPs total
But parameters = only 273M vs 600B!

## Adaptive Compute Flow

```
For each token:
  1. Run 3 iterations (fast probe)
  2. Estimate difficulty from early states
  3. Map difficulty to iteration count:
     - Easy tokens (high confidence): 5 iterations
     - Medium tokens: 20-30 iterations  
     - Hard tokens: 40-50 iterations
  4. Run full iterations
  5. Update running surprisal estimate
```

## Deep Supervision Flow (Training)

```
For each token t:
  For each iteration i:
    logits_i = Head(workspace_i[0])  # First slot → vocab
    loss_i = CrossEntropy(logits_i, target[t])
    weighted_loss += weight[i] * loss_i
  
  total_deep_loss = mean(weighted_loss)
  final_loss += CrossEntropy(logits_N, target[t])
  
Combined loss = 0.6 * final_loss + 0.4 * deep_loss