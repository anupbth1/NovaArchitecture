"""Verify all 3 critical blockers are resolved."""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nova.rcv.brain_cell import BrainCell
from nova.rcv.nova_brain import NovaRCV
from nova.rcv.config import RCVConfig

print("=" * 60)
print("VERIFICATION: 3 Critical Blockers")
print("=" * 60)

print("\n[1/3] Workspace Stability")
cell = BrainCell(d_model=64, expansion=2)
x = torch.randn(2, 8, 64)
norms = [x.norm().item()]
for _ in range(50):
    x = cell(x)
    norms.append(x.norm().item())
ratio = norms[-1] / max(norms[0], 0.01)
stable = norms[-1] < 100 and ratio < 5
print(f"  Norm: {norms[0]:.2f} -> {norms[-1]:.2f} (ratio: {ratio:.2f}x)")
print(f"  Status: {'PASS' if stable else 'FAIL'}")

print("\n[2/3] Training Convergence")
config = RCVConfig(d_model=64, vocab_size=5000, expansion=2, num_slots=8, max_iterations=10)
model = NovaRCV(config)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
losses = []
for step in range(50):
    x = torch.randint(0, 5000, (4, 16))
    loss = model(x, targets=x, iter_limits=10)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    losses.append(loss.item())
converges = losses[-1] < losses[0]
print(f"  Loss: {losses[0]:.3f} -> {losses[-1]:.3f}")
print(f"  Status: {'PASS (converging)' if converges else 'FAIL (not converging)'}")

print("\n[3/3] Gradient Flow")
model.zero_grad()
x = torch.randint(0, 5000, (4, 16))
loss = model(x, targets=x, iter_limits=10)
loss.backward()
nan_count = 0
zero_count = 0
grads = []
for name, p in model.named_parameters():
    if p.grad is not None:
        g = p.grad.norm().item()
        grads.append(g)
        if torch.isnan(p.grad).any():
            nan_count += 1
        if g < 1e-8 and p.numel() > 1:
            zero_count += 1
no_nan = nan_count == 0
ok_ratio = max(grads) / max(min(grads), 1e-10) < 1e6 if grads else True
print(f"  NaN gradients: {nan_count} {'OK' if no_nan else 'FAIL'}")
print(f"  Near-zero gradients: {zero_count}")
print(f"  Max grad: {max(grads):.6f}, Min grad: {min(grads):.6f}")
print(f"  Max/Min ratio: {max(grads)/max(min(grads),1e-8):.1f} {'OK' if ok_ratio else 'FAIL'}")
print(f"  Status: {'PASS' if no_nan and ok_ratio else 'FAIL'}")

print("\n" + "=" * 60)
if stable and converges and no_nan:
    print("ALL 3 CRITICAL BLOCKERS RESOLVED")
else:
    print("SOME BLOCKERS REMAIN")
print("=" * 60)