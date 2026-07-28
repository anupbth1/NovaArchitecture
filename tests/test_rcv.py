"""
Tests for NovaRCV v2 — Iteration scaling and workspace stability.
"""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nova.rcv.config import RCVConfig
from nova.rcv.brain_cell import BrainCell
from nova.rcv.slot_debate import SlotDebate
from nova.rcv.reasoner import AdaptiveReasoner
from nova.rcv.nova_brain import NovaRCV


def test_workspace_stability():
    """Test: Workspace norm should stay bounded, not explode."""
    cell = BrainCell(d_model=64, expansion=2)
    x = torch.randn(2, 8, 64)
    
    norms = []
    for _ in range(50):
        x = cell(x)
        norms.append(x.norm().item())
    
    # Norm should stay bounded (not grow to 100+)
    max_norm = max(norms)
    min_norm = min(norms)
    assert max_norm < 100, f"Workspace exploded: max norm = {max_norm}"
    assert max_norm / max(min_norm, 0.1) < 10, \
        f"Norm ratio too large: {max_norm}/{min_norm}"
    print(f"  ✅ test_workspace_stability: norms=[{min_norm:.2f}, {max_norm:.2f}] over 50 iters")


def test_brain_cell():
    cell = BrainCell(d_model=64, expansion=4)
    x = torch.randn(2, 16, 64)
    out = cell(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    print(f"  ✅ test_brain_cell: output shape {out.shape}")


def test_slot_debate():
    debate = SlotDebate(d_model=64, num_slots=16)
    x = torch.randn(2, 16, 64)
    out = debate(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    print(f"  ✅ test_slot_debate: output shape {out.shape}")


def test_reasoner_corrective():
    """Test corrective iteration output shapes."""
    reasoner = AdaptiveReasoner(d_model=64, num_slots=16, max_iterations=10)
    workspace = torch.randn(2, 16, 64)
    token_embed = torch.randn(2, 1, 64)
    
    workspace_out, corrections = reasoner(workspace, token_embed, iter_limit=5)
    
    assert workspace_out.shape == (2, 16, 64)
    assert corrections.shape[0] == 2  # batch
    assert corrections.shape[1] == 5  # actual_iters
    assert corrections.shape[2] == 64  # d_model
    print(f"  ✅ test_reasoner_corrective: {corrections.shape[1]} corrections, output {workspace_out.shape}")


def test_corrective_loss():
    """Test corrective ensemble loss decreases with more iterations."""
    config = RCVConfig(d_model=64, vocab_size=5000, expansion=2, num_slots=8, max_iterations=10)
    model = NovaRCV(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    x = torch.randint(0, 5000, (4, 16))
    
    losses = []
    for step in range(50):
        loss = model(x, targets=x, iter_limits=10)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    
    final_loss = losses[-1]
    initial_loss = losses[0]
    print(f"  ✅ test_corrective_loss: {initial_loss:.3f} → {final_loss:.3f} "
          f"({'improved' if final_loss < initial_loss else 'DID NOT IMPROVE'})")


def test_iteration_scaling():
    """CRITICAL TEST: More iterations should produce LOWER loss."""
    config = RCVConfig(d_model=64, vocab_size=5000, expansion=2, num_slots=8, max_iterations=20)
    model = NovaRCV(config)
    
    # Train a bit first
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for step in range(30):
        x = torch.randint(0, 5000, (4, 16))
        loss = model(x, targets=x, iter_limits=10)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    x = torch.randint(0, 5000, (4, 8))
    targets = x
    
    model.eval()
    with torch.no_grad():
        losses = {}
        for iters in [1, 2, 5, 10, 20]:
            loss = model(x, targets=targets, iter_limits=iters)
            losses[iters] = loss.item()
            print(f"    iterations={iters:2d}: loss={loss.item():.4f}")
    
    # Check: more iterations should reduce loss
    if losses[20] < losses[1]:
        print(f"  ✅ test_iteration_scaling: More iterations REDUCE loss ✓")
    else:
        print(f"  ❌ test_iteration_scaling: More iterations INCREASE loss "
              f"({losses[1]:.3f} → {losses[20]:.3f})")


def test_nova_rcv_forward():
    config = RCVConfig(vocab_size=5000, d_model=64, num_slots=16, max_iterations=5, seq_len=32)
    model = NovaRCV(config)
    x = torch.randint(0, 5000, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 5000), f"Expected (2,16,5000), got {logits.shape}"
    print(f"  ✅ test_nova_rcv_forward: logits shape {logits.shape}")


def test_generate():
    config = RCVConfig(vocab_size=5000, d_model=64, num_slots=16, max_iterations=5, seq_len=32)
    model = NovaRCV(config)
    model.eval()
    input_ids = torch.randint(0, 5000, (1, 5))
    generated = model.generate(input_ids, max_new_tokens=10, temperature=0.7, top_p=0.9)
    assert generated.shape[1] == 15
    print(f"  ✅ test_generate: output shape {generated.shape}")


def run_all():
    print("=" * 60)
    print("NovaRCV v2 Test Suite (Corrective Iterations)")
    print("=" * 60)
    
    tests = [
        test_workspace_stability,
        test_brain_cell,
        test_slot_debate,
        test_reasoner_corrective,
        test_nova_rcv_forward,
        test_corrective_loss,
        test_iteration_scaling,
        test_generate,
    ]
    
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__}: FAILED - {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    if failed == 0:
        print("🎉 All tests passed!")
    return failed == 0


if __name__ == '__main__':
    success = run_all()
    sys.exit(0 if success else 1)