# NovaArchitecture

## ⚠️ V1 Status: Research Prototype — Frozen

**Nova RCV v1 is now frozen as a research baseline.**

After a complete architecture audit (see `docs/ARCHITECTURE_AUDIT_REPORT.md`), we found:

| Finding | Status |
|---------|--------|
| Training does not converge | ❌ Critical — loss stuck at ~7.0 |
| More iterations increase loss | ❌ Critical — more compute harms |
| "600B" FLOP claim unsupported | ❌ Measured: 6.55B, not 600B |
| Workspace activation explosion | ❌ Norm 0.49 → 14.70 over 10 iters |
| **Architecture novelty** | ⚠️ **Transformer variant (55/100)** |

### Why V1 is being frozen

The architecture contains these Transformer components:
- QKV Attention (SlotDebate)
- MLP + GELU (BrainCell)
- LayerNorm (Pre-norm)
- Residual connections
- Positional encoding
- Learned embeddings

These are rearranged with weight sharing and iterative reuse, but the fundamental computation model is:
**Transformer + Recurrent loop**

This will never produce a "fundamentally new architecture." Maximum outcome: an efficient Transformer variant.

### What V1 accomplished (positive)

1. ✅ Proved that iterative weight reuse can work at the tensor shape level
2. ✅ Identified the exact gradient flow failure point
3. ✅ Built a complete evaluation framework (benchmarks, audit tools)
4. ✅ Documented WHY this approach fails — valuable for future research
5. ✅ Established rigorous evaluation methodology

---

## 🔜 V2: Coming Soon — Zero Transformer Components

V2 will be a **clean-sheet design** with this constraint:

```
ZERO Transformer Components:
❌ No Attention (QKV)
❌ No MLP / FFN blocks
❌ No LayerNorm / RMSNorm
❌ No GELU / ReLU / standard activations
❌ No Residual connections
❌ No Positional encoding
❌ No KV cache
❌ No Learned embeddings as the primary representation
```

### V2 Design Process

1. **Phase 0**: Mathematical specification (in progress)
2. **Phase 1**: Semantic Representation (no token vectors)
3. **Phase 2**: Dynamic State (no fixed-size hidden states)
4. **Phase 3**: Reasoning Runtime (no attention)
5. **Phase 4**: Learning Algorithm (no backprop through static graph)
6. **Phase 5**: Language Generation (no autoregressive token-by-token)
7. **Phase 6**: Multimodal (born multimodal, not adapted)

Each phase requires:
- Mathematical definition of the computation model
- Time complexity analysis
- Memory complexity analysis
- Training algorithm specification
- Proof of why it should scale better than Transformers

---

## 📁 Current Repo Structure (V1 Frozen)

| Directory | Contents |
|-----------|----------|
| `nova/rcv/` | V1 RCV implementation (frozen) |
| `nova/brain/` | Pipeline orchestration |
| `nova/reasoning/` | Hypothesis engine |
| `nova/memory/` | Dynamic graph memory |
| `benchmarks/` | 4 benchmark suites |
| `docs/` | Vision, architecture, audit report |
| `tools/` | Audit & analysis tools |
| `tests/` | 10 unit tests (all passing) |

## 📊 Quick Stats

- **V1 Parameters**: 1.6M (tiny) to 265M (full config)
- **V1 FLOPs/token**: 0.01B to 6.55B
- **V1 Training Status**: ❌ Does not converge
- **V1 Novelty**: 55/100 (Transformer variant)
- **V2 Status**: 🔜 Design phase

## 🤝 License

MIT

---

*"The first version teaches you why you need a second one."*