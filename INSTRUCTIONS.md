# NovaArchitecture RCV - Complete Usage Guide

## 📋 Overview

NovaArchitecture is a Recursive Computation Volume (RCV) language model.  
**Current Status**: ⚠️ Research Prototype - Training convergence issues under investigation.

---

## 1. 🚀 Quick Start

### 1.1 Google Colab Setup

Open this notebook in Colab: [NovaArchitecture Colab Notebook](https://colab.research.google.com/github/anupbth1/NovaArchitecture/blob/main/NovaArchitecture_Colab.ipynb)

Or manually:

```python
# In Colab: Runtime → Change runtime type → T4 GPU (or CPU if no GPU)
# Then run:

# Step 1: Clone the repo
!git clone https://github.com/anupbth1/NovaArchitecture.git
%cd NovaArchitecture

# Step 2: Install dependencies
!pip install torch numpy transformers datasets tqdm

# Step 3: Run tests
!python tests/test_rcv.py

# Step 4: Run demo
!python examples/demo.py

# Step 5: Run architecture audit
!python tools/audit.py

# Step 6: Run benchmarks (dry run)
!python benchmarks/run_benchmarks.py --dry-run
```

### 1.2 Local Setup

```bash
# Prerequisites: Python 3.10+, pip, git

# Clone
git clone https://github.com/anupbth1/NovaArchitecture.git
cd NovaArchitecture

# Recommended: Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install
pip install -r nova/requirements.txt
```

---

## 2. 📁 Project Structure

```
NovaArchitecture/
├── nova/                      # Main package
│   ├── __init__.py           # Package exports
│   ├── rcv/                  # CORE: RCV implementation
│   │   ├── brain_cell.py     # The knowledge MLP (reused iteratively)
│   │   ├── slot_debate.py    # Multi-perspective slot attention
│   │   ├── reasoner.py       # Iterative reasoning loop
│   │   ├── nova_brain.py     # Complete RCV model
│   │   ├── config.py         # Configuration
│   │   └── trainer.py        # Training pipeline
│   ├── brain/                # Pipeline orchestration
│   ├── reasoning/            # Hypothesis generation/verification
│   ├── memory/               # Dynamic knowledge graph
│   ├── planner/              # Goal planning
│   ├── core/                 # Low-level data structures
│   ├── representation/       # Semantic units & tokenization
│   └── workers/              # Parallel execution
├── benchmarks/               # Evaluation benchmarks
│   ├── coding_benchmark.py   # Code generation tests (9 cases)
│   ├── reasoning_benchmark.py# Logic/math tests (9 cases)
│   ├── debugging_benchmark.py# Bug finding tests (6 cases)
│   ├── knowledge_benchmark.py# Factual knowledge tests (6 cases)
│   └── run_benchmarks.py     # Benchmark runner
├── docs/                     # Documentation
│   ├── 000_Project_Vision.md
│   ├── 001_Requirements.md
│   ├── 002_Architecture.md
│   ├── 003_DataFlow.md
│   ├── 004_Mathematics.md
│   ├── Decisions.md
│   └── ARCHITECTURE_AUDIT_REPORT.md  # Latest audit
├── examples/
│   └── demo.py               # Full demo script
├── tests/
│   └── test_rcv.py           # 10 unit tests
├── tools/
│   ├── audit.py              # Architecture audit tool
│   └── iteration_scaling_test.py  # Training convergence test
├── INSTRUCTIONS.md           # This file
└── nova/requirements.txt     # Python dependencies
```

---

## 3. 🧪 Running Tests

```bash
# Run all 10 unit tests
python tests/test_rcv.py

# Expected output:
# ============================================================
# NovaRCV Test Suite
# ============================================================
#   ✅ test_config: param_count=1,509,888
#   ✅ test_brain_cell: output shape torch.Size([2, 16, 64])
#   ✅ test_slot_debate: output shape torch.Size([2, 16, 64])
#   ✅ test_reasoner: iterations=5, output_shape=torch.Size([2, 16, 64])
#   ✅ test_reasoner_deep_supervision: loss=3.076
#   ✅ test_nova_rcv_forward: logits shape torch.Size([2, 16, 5000])
#   ✅ test_nova_rcv_training: loss=7.005
#   ✅ test_nova_rcv_param_count: total=722,316, effective=3.6M
#   ✅ test_generate: output shape torch.Size([1, 15])
#   ✅ test_adaptive_iter_limits: default=6.974, high=6.974, low=5.608
# ============================================================
# Results: 10/10 passed, 0 failed
```

---

## 4. 🎯 Running Demo

```bash
# Run the complete demo
python examples/demo.py

# This shows:
# - Model creation with parameter counts
# - Training on synthetic data (20 steps)
# - Adaptive compute demonstration
# - Theoretical analysis vs Transformers
```

---

## 5. 📊 Running Benchmarks

```bash
# Dry run (just shows structure, no model needed)
python benchmarks/run_benchmarks.py --dry-run

# Run all benchmarks with a model
python benchmarks/run_benchmarks.py --train

# Run specific benchmarks
python benchmarks/run_benchmarks.py --benchmarks coding,reasoning

# Output:
#   coding: 9 tests (Fibonacci, FizzBuzz, Binary Search, etc.)
#   reasoning: 9 tests (Logic, Math, Planning, etc.)
#   debugging: 6 tests (Bug finding/fixing)
#   knowledge: 6 tests (Facts)
```

---

## 6. 🔍 Running Architecture Audit

```bash
# Comprehensive architecture analysis
python tools/audit.py

# This measures:
# - Parameter counts and FLOPs
# - Gradient flow analysis
# - Inference and training speed
# - Transformer-likeness score
# - All detected issues

# Iteration scaling test (deep investigation)
python tools/iteration_scaling_test.py
```

---

## 7. ⚙️ Configuration Options

Edit `nova/rcv/config.py` to change model scale:

```python
@dataclass
class RCVConfig:
    # Model size
    d_model: int = 2048        # Hidden dimension (128=tiny, 2048=full)
    vocab_size: int = 50257    # Vocabulary (5000=tiny, 50257=GPT-2)
    num_slots: int = 64        # Memory slots (8=tiny, 64=full)
    expansion: int = 4         # MLP expansion (2=tiny, 4-8=full)
    max_iterations: int = 30   # Iterations per token (10=tiny, 30-50=full)
    
    # Training
    batch_size: int = 4
    seq_len: int = 256
    lr: float = 3e-4
    total_steps: int = 100000
    
    # Adaptive compute
    min_iter: int = 5          # Minimum iterations for easy tokens
    max_iter_cap: int = 50     # Maximum iterations for hard tokens
```

### Recommended Configurations

| Config | Params | FLOPs/token | Use Case |
|--------|--------|-------------|----------|
| Tiny (d=128, iter=10) | 1.6M | 0.01B | Testing on CPU |
| Small (d=512, iter=15) | 17M | 0.25B | Colab free GPU |
| Medium (d=1024, iter=20) | 118M | 1.43B | 8GB GPU |
| Full (d=2048, iter=30) | 265M | 6.55B | 12GB+ GPU |
| Large (d=3072, iter=50) | ~800M | ~20B | 24GB+ GPU |

---

## 8. 💻 Colab Testing Instructions (Step-by-Step)

### Google Colab Link: https://colab.research.google.com/

```python
# ───────────────────────────────────────────
# CELL 1: Clone and setup (run this first)
# ───────────────────────────────────────────
!git clone https://github.com/anupbth1/NovaArchitecture.git
%cd NovaArchitecture

# ───────────────────────────────────────────
# CELL 2: Install dependencies
# ───────────────────────────────────────────
!pip install torch numpy transformers datasets tqdm

# ───────────────────────────────────────────
# CELL 3: Verify installation
# ───────────────────────────────────────────
!python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# ───────────────────────────────────────────
# CELL 4: Run tests
# ───────────────────────────────────────────
!python tests/test_rcv.py

# ───────────────────────────────────────────
# CELL 5: Run demo
# ───────────────────────────────────────────
!python examples/demo.py

# ───────────────────────────────────────────
# CELL 6: Run architecture audit
# ───────────────────────────────────────────
!python tools/audit.py

# ───────────────────────────────────────────
# CELL 7: Run iteration scaling test
# ───────────────────────────────────────────
!python tools/iteration_scaling_test.py

# ───────────────────────────────────────────
# CELL 8: Run benchmarks (dry run)
# ───────────────────────────────────────────
!python benchmarks/run_benchmarks.py --dry-run

# ───────────────────────────────────────────
# CELL 9 (Optional): Train small model
# ───────────────────────────────────────────
import sys, torch
sys.path.insert(0, '.')
from nova.rcv.nova_brain import NovaRCV
from nova.rcv.config import RCVConfig

config = RCVConfig(d_model=128, vocab_size=5000, expansion=2, num_slots=8, max_iterations=10)
model = NovaRCV(config)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for step in range(50):
    x = torch.randint(0, 5000, (4, 32))
    loss = model(x, targets=x)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 10 == 0:
        print(f"Step {step}: loss={loss.item():.4f}")
```

---

## 9. ⚡ Known Issues & Current Limitations

| Issue | Status | Impact |
|-------|--------|--------|
| Training does not converge | 🚧 Under investigation | Model cannot learn from data |
| More iterations = worse loss | 🚧 Under investigation | Core RCV premise broken |
| Workspace activation explosion | 🚧 Under investigation | Unbounded norm growth |
| "600B" FLOP claim invalid | ✅ Documented | Marketing claim, not technical |
| Sequential token processing | ⚠️ Architectural | O(T) latency vs Transformer O(1) |
| Position encoding limited length | ⚠️ Known | Generation degrades after seq_len |

See `docs/ARCHITECTURE_AUDIT_REPORT.md` for full details.

---

## 10. 🛠️ Development

### Adding new features

```python
# Create new module
touch nova/myfeature/__init__.py

# Add to imports in nova/__init__.py
from .myfeature import MyFeature

# Create tests
touch tests/test_myfeature.py

# Create benchmarks
touch benchmarks/myfeature_benchmark.py

# Add to benchmark runner
# Edit benchmarks/run_benchmarks.py
```

### Running specific file

```bash
# From project root
python -m nova.rcv.nova_brain    # Module import
python tests/test_rcv.py          # Direct script
python -m pytest tests/           # Using pytest
```

---

## 11. 📦 Dependencies

```
torch>=2.0.0          # Core tensor operations
numpy>=1.21.0         # Numerical operations
datasets>=2.0.0       # For loading TinyStories (optional)
transformers>=4.30.0  # For tokenizer (optional)
tqdm>=4.64.0          # Progress bars (optional)
```

Install: `pip install -r nova/requirements.txt`

---

## 12. 📝 License

MIT License - See LICENSE file

---

## 13. 🙋 Support

- GitHub Issues: https://github.com/anupbth1/NovaArchitecture/issues
- Author: anupbth1

---

*Last updated: 2026-07-28*