"""
Tests for NovaRCV - Recursive Computation Volume LLM
=====================================================
"""
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nova.rcv.config import RCVConfig
from nova.rcv.brain_cell import BrainCell
from nova.rcv.slot_debate import SlotDebate
from nova.rcv.reasoner import AdaptiveReasoner
from nova.rcv.nova_brain import NovaRCV


def test_config():
    """Test RCVConfig parameter calculations."""
    config = RCVConfig(d_model=128, vocab_size=5000, expansion=4)
    assert config.d_model == 128
    assert config.vocab_size == 5000
    assert config.max_iterations == 30
    assert config.effective_depth == 30
    params = config.param_count()
    assert params > 0, "Parameter count should be positive"
    print(f"  ✅ test_config: param_count={params:,}")


def test_brain_cell():
    """Test BrainCell forward pass."""
    cell = BrainCell(d_model=64, expansion=4)
    x = torch.randn(2, 16, 64)  # (B, M, D)
    out = cell(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    assert not torch.isnan(out).any(), "Output has NaN values"
    print(f"  ✅ test_brain_cell: output shape {out.shape}")


def test_slot_debate():
    """Test SlotDebate communication among slots."""
    debate = SlotDebate(d_model=64, num_slots=16)
    x = torch.randn(2, 16, 64)  # (B, M, D)
    out = debate(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    assert not torch.isnan(out).any(), "Output has NaN values"
    print(f"  ✅ test_slot_debate: output shape {out.shape}")


def test_reasoner():
    """Test AdaptiveReasoner iterative refinement."""
    reasoner = AdaptiveReasoner(d_model=64, num_slots=16, max_iterations=5)
    workspace = torch.randn(2, 16, 64)
    token_embed = torch.randn(2, 1, 64)
    
    workspace_out, iter_outputs = reasoner(workspace, token_embed, iter_limit=5)
    assert workspace_out.shape == (2, 16, 64)
    assert iter_outputs.shape[0] == 2  # batch
    assert iter_outputs.shape[2] == 64  # d_model
    print(f"  ✅ test_reasoner: iterations={iter_outputs.shape[1]}, output_shape={workspace_out.shape}")


def test_reasoner_deep_supervision():
    """Test deep supervision loss computation."""
    reasoner = AdaptiveReasoner(d_model=64, num_slots=16, max_iterations=5)
    # Simulate logits from 3 iterations
    all_iter_logits = torch.randn(2, 3, 5000)  # (B, iters, vocab)
    targets = torch.randint(0, 5000, (2,))
    
    loss = reasoner.compute_deep_supervision_loss(all_iter_logits, targets)
    assert loss.item() > 0, "Loss should be positive"
    print(f"  ✅ test_reasoner_deep_supervision: loss={loss.item():.3f}")


def test_nova_rcv_forward():
    """Test NovaRCV forward pass."""
    config = RCVConfig(
        vocab_size=5000,
        d_model=64,
        num_slots=16,
        max_iterations=5,
        seq_len=32,
    )
    model = NovaRCV(config)
    
    x = torch.randint(0, 5000, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 5000), f"Expected (2,16,5000), got {logits.shape}"
    print(f"  ✅ test_nova_rcv_forward: logits shape {logits.shape}")


def test_nova_rcv_training():
    """Test NovaRCV training step with deep supervision."""
    config = RCVConfig(
        vocab_size=5000,
        d_model=64,
        num_slots=16,
        max_iterations=5,
        seq_len=32,
    )
    model = NovaRCV(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training step
    x = torch.randint(0, 5000, (2, 16))
    loss = model(x, targets=x)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    assert loss.item() > 0, "Loss should be positive"
    print(f"  ✅ test_nova_rcv_training: loss={loss.item():.3f}")


def test_nova_rcv_param_count():
    """Test parameter counting."""
    config = RCVConfig(
        vocab_size=5000,
        d_model=64,
        num_slots=16,
        max_iterations=5,
    )
    model = NovaRCV(config)
    counts = model.get_param_count()
    
    assert 'total' in counts
    assert counts['total'] > 0
    assert 'effective_compute_depth' in counts
    assert counts['effective_compute_depth'] == 5
    
    effective = model.get_effective_size()
    assert 'B' in effective or 'M' in effective
    print(f"  ✅ test_nova_rcv_param_count: total={counts['total']:,}, effective={effective}")


def test_generate():
    """Test text generation."""
    config = RCVConfig(
        vocab_size=5000,
        d_model=64,
        num_slots=16,
        max_iterations=5,
        seq_len=32,
    )
    model = NovaRCV(config)
    model.eval()
    
    input_ids = torch.randint(0, 5000, (1, 5))
    generated = model.generate(
        input_ids,
        max_new_tokens=10,
        temperature=0.7,
        top_p=0.9,
    )
    assert generated.shape[1] == 15  # 5 prompt + 10 new
    print(f"  ✅ test_generate: output shape {generated.shape}")


def test_adaptive_iter_limits():
    """Test adaptive compute with different iteration limits."""
    config = RCVConfig(
        vocab_size=5000,
        d_model=64,
        num_slots=16,
        max_iterations=10,
        seq_len=16,
    )
    model = NovaRCV(config)
    
    x = torch.randint(0, 5000, (2, 8))
    
    # Run with different limits
    loss_default = model(x, targets=x)
    loss_high = model(x, targets=x, iter_limits=20)
    loss_low = model(x, targets=x, iter_limits=2)
    
    print(f"  ✅ test_adaptive_iter_limits: default={loss_default.item():.3f}, high={loss_high.item():.3f}, low={loss_low.item():.3f}")


def run_all():
    """Run all tests."""
    print("=" * 60)
    print("NovaRCV Test Suite")
    print("=" * 60)
    
    tests = [
        test_config,
        test_brain_cell,
        test_slot_debate,
        test_reasoner,
        test_reasoner_deep_supervision,
        test_nova_rcv_forward,
        test_nova_rcv_training,
        test_nova_rcv_param_count,
        test_generate,
        test_adaptive_iter_limits,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__}: FAILED - {e}")
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    if failed == 0:
        print("🎉 All tests passed!")
    print(f"{'=' * 60}")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all()
    sys.exit(0 if success else 1)