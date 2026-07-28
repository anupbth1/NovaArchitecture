# NovaArchitecture — RCV Language Model

## Project Name
**NovaArchitecture** — A Recursive Computation Volume (RCV) Language Model

## Mission
Build a **1-billion parameter** LLM that achieves **600-billion+ parameter** accuracy through **iterative reasoning** instead of **stacking layers**.

## Goals

| Goal | Target | Why |
|------|--------|-----|
| Training Parameters | **1B parameters** | Fits on a single GPU for training |
| Effective Compute Depth | **600B+ equivalent** | 30-50x iteration reuse × 64 slots |
| Reasoning Quality | **GPT-4 level** | Coding, debugging, math, planning |
| Inference Cost | **1B model cost** | Same FLOPs as 1B model inference |
| Training Cost | **<$10K** | Single GPU, no cluster needed |
| Architecture | **No MoE** | Single dense brain, iteratively reused |

## Non-Goals
- **No Mixture of Experts (MoE)** — MoE increases total params but doesn't increase compute per token
- **No massive KV-cache** — We don't attend to past tokens, we compress into workspace slots
- **No parallel branches** — Single sequential pipeline, no routing overhead
- **No GPU cluster** — Designed to train and run on a single GPU

## Architecture Philosophy
**"One brain, many thoughts."** Instead of stacking 100 layers (like GPT-4), we have ONE dense layer that we iterate 50 times per token. Each iteration refines the model's understanding. This is like having 1 expert who thinks for 50 minutes vs 50 experts who each think for 1 minute. The single expert is cheaper and often more coherent.

## Research Rules
1. **Iterative over parallel** — Always prefer reusing a small model many times over creating a large model once
2. **Compute over parameters** — FLOPs matter more than parameter count for quality
3. **Dense over sparse** — Single dense expert beats mixture of sparse experts
4. **Workspace over cache** — Compress history into slots rather than caching raw tokens
5. **Deep supervision over end-only** — Train every iteration, not just the final one
6. **Adaptive over fixed** — Harder tokens should get more compute
7. **Stability over speed** — Use LayerNorm, scaled residuals, gradient clipping

## Success Criteria

### Phase 1: Proof (Current)
- [x] Core RCV forward pass works (NovaRCV)
- [x] Deep supervision training works
- [x] Adaptive compute works
- [x] All forward/backward shapes correct
- [x] Parameter counting matches expectations

### Phase 2: Training (Next)
- [ ] Train on TinyStories (1B tokens)
- [ ] Achieve <4.0 perplexity
- [ ] Generate coherent stories
- [ ] Pass coding benchmarks >60%

### Phase 3: Scale
- [ ] Train d_model=2048, expansion=8 (~800M params)
- [ ] Train on Python code (TheStack)
- [ ] Achieve HumanEval >30%
- [ ] Achieve GSM8K >40%

### Phase 4: Production
- [ ] Train to full 1B params
- [ ] Deploy as API
- [ ] Fine-tune for specific tasks
- [ ] Continual learning integration