#!/usr/bin/env python3
"""
Iteration Scaling Investigation
================================
Tests whether more iterations actually help (decrease loss).
If they don't, the entire RCV premise is broken.
"""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nova.rcv.nova_brain import NovaRCV
from nova.rcv.config import RCVConfig

torch.manual_seed(42)

def test_iteration_scaling_untrained():
    """Test: In an UNTRAINED model, do more iterations help?"""
    print("=" * 60)
    print("TEST 1: Untrained Model - Iteration Scaling")
    print("=" * 60)
    
    config = RCVConfig(d_model=64, vocab_size=5000, expansion=2, num_slots=8, max_iterations=20)
    model = NovaRCV(config)
    
    x = torch.randint(0, 5000, (4, 16))
    targets = x  # next token prediction (same shift inside model)
    
    with torch.no_grad():
        for iters in [1, 2, 5, 10, 20]:
            loss = model(x, targets=targets, iter_limits=iters)
            print(f"  iterations={iters:2d}: loss={loss.item():.4f}")
    
    print("  > For untrained model, MORE iterations = MORE random drift = HIGHER loss")
    print("  > This is NORMAL. Training should fix this.")
    print()


def test_training_effectiveness():
    """Test: Does training with deep supervision actually reduce loss?"""
    print("=" * 60)
    print("TEST 2: Training Effectiveness")
    print("=" * 60)
    
    config = RCVConfig(d_model=64, vocab_size=5000, expansion=2, num_slots=8, max_iterations=10)
    model = NovaRCV(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    losses = []
    for step in range(100):
        x = torch.randint(0, 5000, (4, 32))
        loss = model(x, targets=x, iter_limits=10)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if step % 20 == 0:
            print(f"  Step {step:3d}: loss={loss.item():.4f}")
    
    print(f"  Final loss: {losses[-1]:.4f} (started at {losses[0]:.4f})")
    
    if losses[-1] > losses[0]:
        print("  ❌ CRITICAL: Training is NOT reducing loss!")
    else:
        print(f"  ✅ Training reduced loss by {losses[0] - losses[-1]:.4f}")
    
    return model


def test_iteration_scaling_trained(model):
    """Test: After training, do more iterations help?"""
    print()
    print("=" * 60)
    print("TEST 3: Trained Model - Iteration Scaling")
    print("=" * 60)
    
    x = torch.randint(0, 5000, (4, 16))
    targets = x
    
    model.eval()
    with torch.no_grad():
        for iters in [1, 2, 5, 10, 15, 20]:
            loss = model(x, targets=targets, iter_limits=iters)
            print(f"  iterations={iters:2d}: loss={loss.item():.4f}")
    
    print("  > If loss DECREASES with more iterations: RCV works")
    print("  > If loss INCREASES with more iterations: RCV broken (more compute harms)")


def test_deep_supervision_vs_final():
    """Test: Is deep supervision actually doing anything?"""
    print()
    print("=" * 60)
    print("TEST 4: Deep Supervision vs Final-Only Loss")
    print("=" * 60)
    
    config = RCVConfig(
        d_model=64, vocab_size=5000, expansion=2, num_slots=8, 
        max_iterations=10,
        final_loss_weight=1.0,  # Deep supervision disabled
        deep_loss_weight=0.0,
    )
    model_no_deep = NovaRCV(config)
    
    config2 = RCVConfig(
        d_model=64, vocab_size=5000, expansion=2, num_slots=8, 
        max_iterations=10,
        final_loss_weight=0.6,
        deep_loss_weight=0.4,
    )
    model_deep = NovaRCV(config2)
    
    # Train both
    x = torch.randint(0, 5000, (4, 32))
    
    opt_no_deep = torch.optim.AdamW(model_no_deep.parameters(), lr=3e-4)
    opt_deep = torch.optim.AdamW(model_deep.parameters(), lr=3e-4)
    
    for step in range(50):
        xb = torch.randint(0, 5000, (4, 32))
        
        loss_no_deep = model_no_deep(xb, targets=xb, iter_limits=10)
        opt_no_deep.zero_grad()
        loss_no_deep.backward()
        torch.nn.utils.clip_grad_norm_(model_no_deep.parameters(), 1.0)
        opt_no_deep.step()
        
        loss_deep = model_deep(xb, targets=xb, iter_limits=10)
        opt_deep.zero_grad()
        loss_deep.backward()
        torch.nn.utils.clip_grad_norm_(model_deep.parameters(), 1.0)
        opt_deep.step()
        
        if step % 20 == 0:
            print(f"  Step {step}: no_deep={loss_no_deep.item():.4f}, deep={loss_deep.item():.4f}")
    
    # Test both on iteration scaling
    xt = torch.randint(0, 5000, (4, 16))
    targets = xt
    
    model_no_deep.eval()
    model_deep.eval()
    
    print("\nIteration scaling comparison:")
    with torch.no_grad():
        for iters in [1, 3, 5, 10]:
            l1 = model_no_deep(xt, targets=targets, iter_limits=iters)
            l2 = model_deep(xt, targets=targets, iter_limits=iters)
            trend1 = "↓" if (iters == 10 and l1.item() < model_no_deep(xt, targets=targets, iter_limits=1).item()) else "↑"
            trend2 = "↓" if (iters == 10 and l2.item() < model_deep(xt, targets=targets, iter_limits=1).item()) else "↑"
            print(f"  iters={iters:2d}: no_deep={l1.item():.4f}, deep={l2.item():.4f}")


def test_workspace_drift():
    """Test: Does the workspace actually improve with iterations, or just drift?"""
    print()
    print("=" * 60)
    print("TEST 5: Workspace Drift Analysis")
    print("=" * 60)
    
    config = RCVConfig(d_model=64, vocab_size=5000, expansion=2, num_slots=8, max_iterations=10)
    model = NovaRCV(config)
    
    x = torch.randint(0, 5000, (1, 4))
    workspace = model.init_workspace.expand(1, -1, -1)
    
    # Track workspace changes per iteration for the first token
    token_embed = model.embed(x[:, 0:1]) * model.embed_scale
    pos = torch.tensor([[0]])
    token_embed = token_embed + model.pos_embed(pos)
    
    print("Workspace norm changes across iterations for first token:")
    with torch.no_grad():
        # Mix token
        B, M, D = workspace.shape
        token_expanded = token_embed.expand(-1, M, -1)
        combined = torch.cat([workspace, token_expanded], dim=-1)
        workspace = model.reasoner.token_mixer(combined) + workspace
        
        initial_norm = workspace.norm().item()
        print(f"    Initial workspace norm: {initial_norm:.4f}")
        
        for t in range(10):
            workspace_before = workspace.clone()
            workspace = model.reasoner.brain(workspace)
            workspace = model.reasoner.debate(workspace)
            
            norm = workspace.norm().item()
            diff = (workspace - workspace_before).norm().item()
            delta = diff / norm if norm > 0 else 0
            
            print(f"    Iter {t+1:2d}: norm={norm:.4f}, delta={delta:.4f} ({'stable' if 0.001 < delta < 2.0 else 'DIVERGING' if delta > 2.0 else 'STAGNATING' if delta < 0.001 else 'OK'})")
    
    # Check: does the first slot actually get better?
    print("\nFirst slot projection through head:")
    with torch.no_grad():
        final_state = workspace[:, 0, :]
        logits = model.head(model.ln_out(final_state))
        probs = torch.softmax(logits, dim=-1)
        top5 = torch.topk(probs, 5)
        print(f"    Top-5 token IDs: {top5.indices[0].tolist()}")
        print(f"    Top-5 probs: {[f'{p:.4f}' for p in top5.values[0].tolist()]}")
    
    return workspace


if __name__ == '__main__':
    test_iteration_scaling_untrained()
    trained_model = test_training_effectiveness()
    test_iteration_scaling_trained(trained_model)
    test_deep_supervision_vs_final()
    test_workspace_drift()