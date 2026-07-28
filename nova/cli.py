#!/usr/bin/env python3
"""
NovaRCV CLI - Complete Command Line Interface
===============================================
Ek hi command se sab kuch: train, convert, test, inference, benchmark.

Usage:
    # Train a new model
    python -m nova.cli train --model tiny --data my_text.txt --steps 1000
    
    # Train with specific parameters
    python -m nova.cli train --d-model 512 --vocab-size 32000 --slots 32 \
        --iterations 10 --batch-size 8 --seq-len 128 --steps 5000 --device cuda
    
    # Convert existing model to RCV
    python -m nova.cli convert --from gpt2 --save model.pt
    
    # Convert and train
    python -m nova.cli convert --from gpt2 --save model.pt --train --steps 500
    
    # Run inference
    python -m nova.cli generate --model model.pt --prompt "Hello world" --tokens 50
    
    # Run all tests
    python -m nova.cli test
    
    # Run benchmarks
    python -m nova.cli benchmark --dry-run
    
    # Full architecture audit
    python -m nova.cli audit
"""
import sys, os, argparse, torch, json, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nova.rcv.nova_brain import NovaRCV
from nova.rcv.config import RCVConfig


# ============================================================
# COMMANDS
# ============================================================

def cmd_train(args):
    """Train a new NovaRCV model from scratch."""
    # Build config from CLI args
    if args.mode and args.mode in ['tiny', 'small', 'medium', 'full_1b']:
        presets = {
            'tiny':    RCVConfig(d_model=128,  vocab_size=5000,  num_slots=8,   max_iterations=5,  expansion=2, batch_size=4,  seq_len=64,   lr=3e-4,   total_steps=1000,  min_iter=2,  max_iter_cap=10),
            'small':   RCVConfig(d_model=512,  vocab_size=32000, num_slots=32,  max_iterations=10, expansion=4, batch_size=8,  seq_len=128,  lr=3e-4,   total_steps=5000,  min_iter=3,  max_iter_cap=20),
            'medium':  RCVConfig(d_model=1024, vocab_size=50257, num_slots=64,  max_iterations=15, expansion=4, batch_size=4,  seq_len=256,  lr=2e-4,   total_steps=50000, min_iter=3,  max_iter_cap=25),
            'full_1b': RCVConfig(d_model=2048, vocab_size=50257, num_slots=64,  max_iterations=30, expansion=4, batch_size=2,  seq_len=512,  lr=1.5e-4, total_steps=200000, min_iter=5,  max_iter_cap=50),
        }
        cfg = presets[args.mode]
    else:
        cfg = RCVConfig(
            d_model=args.d_model or 128,
            vocab_size=args.vocab_size or 5000,
            num_slots=args.slots or 8,
            max_iterations=args.iterations or 5,
            expansion=args.expansion or 2,
            batch_size=args.batch_size or 4,
            seq_len=args.seq_len or 64,
            lr=args.lr or 3e-4,
            total_steps=args.steps or 1000,
            min_iter=args.min_iter or 2,
            max_iter_cap=args.max_iter_cap or 10,
        )
    
    if args.steps:
        cfg.total_steps = args.steps
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = args.save or f"checkpoints/nova_rcv_{args.mode or 'custom'}_final.pt"
    
    print(f"""
╔══════════════════════════════════════════════╗
║     NovaRCV Training                         ║
╠══════════════════════════════════════════════╣
║  d_model={cfg.d_model:<5}  vocab={cfg.vocab_size:<6}  slots={cfg.num_slots:<4}  ║
║  iterations={cfg.max_iterations:<4}  seq_len={cfg.seq_len:<5}  batch={cfg.batch_size:<4}  ║
║  steps={cfg.total_steps:<6}  device={device:<8}      ║
╚══════════════════════════════════════════════╝""")
    
    # Create model
    model = NovaRCV(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Load data (or use synthetic)
    data = None
    if args.data and os.path.exists(args.data):
        with open(args.data, 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        char_to_idx = {c: i for i, c in enumerate(chars)}
        data = torch.tensor([char_to_idx.get(c, 0) for c in text], dtype=torch.long)
        print(f"  Data: {len(text)} chars, {len(chars)} unique")
    if data is None:
        print(f"  No data file. Using synthetic random data.")
        data = torch.randint(0, min(cfg.vocab_size, 100), (10000,))
    
    # Resume?
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_step = checkpoint.get('step', 0)
        print(f"  Resumed from step {start_step}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.total_steps, eta_min=cfg.lr*0.1)
    
    # Training loop
    print(f"\n  Training {cfg.total_steps} steps...")
    model.train()
    start_time = time.time()
    
    for step in range(start_step, cfg.total_steps):
        # Get random batch
        idx = torch.randint(0, len(data) - cfg.seq_len - 1, (cfg.batch_size,))
        x = torch.stack([data[i:i+cfg.seq_len] for i in idx]).to(device)
        targets = x[:, 1:].contiguous()
        inputs = x[:, :-1].contiguous()
        
        loss = model(inputs, targets=targets, iter_limits=cfg.max_iterations)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if step % 50 == 0:
            elapsed = time.time() - start_time
            tok_s = cfg.batch_size * cfg.seq_len * 50 / max(elapsed, 0.1)
            lr_val = scheduler.get_last_lr()[0]
            print(f"  Step {step:6d}/{cfg.total_steps} | Loss: {loss.item():.4f} | LR: {lr_val:.2e} | {tok_s:.0f} tok/s")
            start_time = time.time()
        
        if step > 0 and step % 500 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({'step': step, 'model_state_dict': model.state_dict(), 'config': cfg}, f"checkpoints/step_{step}.pt")
            print(f"  💾 Saved checkpoint at step {step}")
    
    # Save final
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    torch.save({'step': cfg.total_steps, 'model_state_dict': model.state_dict(), 'config': cfg}, save_path)
    print(f"\n✅ Model saved to {save_path}")
    print(f"   Parameters: {total_params:,}")


def cmd_convert(args):
    """Convert an existing HuggingFace model to RCV."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("❌ transformers not installed. Run: pip install transformers")
        return
    
    device = args.device or 'cpu'
    print(f"\n  Converting {args.source} → {args.save or 'rcv_converted.pt'}")
    
    # Load source
    print(f"  Loading {args.source}...")
    source = AutoModelForCausalLM.from_pretrained(args.source, torch_dtype=torch.float32)
    src_sd = source.state_dict()
    src_params = sum(p.numel() for p in source.parameters())
    
    # Detect dimensions
    d_model = None
    for k, v in src_sd.items():
        if 'embed' in k.lower() and v.dim() == 2:
            d_model = v.shape[1]
            vocab_size = v.shape[0]
            break
        if 'wte' in k.lower():
            d_model = v.shape[1]
            vocab_size = v.shape[0]
            break
    if not d_model:
        d_model = 768
        vocab_size = 50257
    
    # Find MLP dimensions
    expansion = 4
    for k, v in src_sd.items():
        if 'mlp' in k.lower() and 'c_fc' in k.lower() and v.dim() == 2:
            expansion = v.shape[0] // d_model
            break
        if 'mlp' in k.lower() and 'gate_proj' in k.lower() and v.dim() == 2:
            expansion = v.shape[0] // d_model
            break
    
    print(f"  Detected: d_model={d_model}, vocab={vocab_size}, expansion={expansion}")
    
    # Create RCV
    cfg = RCVConfig(d_model=d_model, vocab_size=vocab_size, expansion=max(2, expansion), 
                     num_slots=64, max_iterations=30, seq_len=2048)
    model = NovaRCV(cfg)
    rcv_sd = model.state_dict()
    
    # Map weights
    mapped = 0
    # Embedding
    for k, v in src_sd.items():
        if ('embed' in k.lower() or 'wte' in k.lower()) and v.dim() == 2:
            if 'embed.weight' in rcv_sd:
                d = min(v.shape[0], rcv_sd['embed.weight'].shape[0])
                rcv_sd['embed.weight'][:d] = v[:d].to(rcv_sd['embed.weight'].dtype)
                mapped += 1
                print(f"  ✅ Embedding: {v.shape}")
    
    # Head
    for k, v in src_sd.items():
        if 'lm_head' in k.lower() and v.dim() == 2:
            if 'head.weight' in rcv_sd:
                d = min(v.shape[0], rcv_sd['head.weight'].shape[0])
                rcv_sd['head.weight'][:d] = v[:d].to(rcv_sd['head.weight'].dtype)
                mapped += 1
                print(f"  ✅ Head: {v.shape}")
    
    # MLP layers (average all)
    fc1_list, fc2_list = [], []
    for k, v in src_sd.items():
        if k.endswith('.weight') and 'mlp' in k.lower():
            if any(x in k for x in ['c_fc', 'gate_proj', 'up_proj', 'dense_h_to_4h']):
                fc1_list.append(v.to(torch.float32))
            elif any(x in k for x in ['c_proj', 'down_proj', 'dense_4h_to_h']):
                fc2_list.append(v.to(torch.float32))
    
    if fc1_list and 'reasoner.brain.fc1.weight' in rcv_sd:
        avg_fc1 = sum(fc1_list) / len(fc1_list)
        avg_fc2 = sum(fc2_list) / len(fc2_list)
        d1 = min(avg_fc1.shape[0], rcv_sd['reasoner.brain.fc1.weight'].shape[0])
        d2 = min(avg_fc1.shape[1], rcv_sd['reasoner.brain.fc1.weight'].shape[1])
        rcv_sd['reasoner.brain.fc1.weight'][:d1, :d2] = avg_fc1[:d1, :d2].to(rcv_sd['reasoner.brain.fc1.weight'].dtype)
        d1 = min(avg_fc2.shape[0], rcv_sd['reasoner.brain.fc2.weight'].shape[0])
        d2 = min(avg_fc2.shape[1], rcv_sd['reasoner.brain.fc2.weight'].shape[1])
        rcv_sd['reasoner.brain.fc2.weight'][:d1, :d2] = avg_fc2[:d1, :d2].to(rcv_sd['reasoner.brain.fc2.weight'].dtype)
        mapped += 2
        print(f"  ✅ BrainCell: {avg_fc1.shape} (avg of {len(fc1_list)} layers)")
    
    # LayerNorms
    for k, v in src_sd.items():
        if 'norm' in k.lower() and v.dim() == 1 and ('h.0' in k or 'layers.0' in k or k.startswith('transformer.h.0')):
            for rcv_k in ['reasoner.brain.norm1.weight', 'reasoner.brain.norm2.weight',
                         'reasoner.iter_norm.weight', 'ln_out.weight']:
                if rcv_k in rcv_sd and rcv_sd[rcv_k].shape == v.shape:
                    rcv_sd[rcv_k] = v.to(rcv_sd[rcv_k].dtype)
                    mapped += 1
                    print(f"  ✅ Norm: {k} → {rcv_k}")
                    break
    
    model.load_state_dict(rcv_sd, strict=False)
    save_path = args.save or 'checkpoints/rcv_converted.pt'
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    torch.save({'source': args.source, 'model_state_dict': model.state_dict(), 'config': cfg}, save_path)
    
    rcv_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Converted: {args.source} ({src_params:,} params) → RCV ({rcv_params:,} params)")
    print(f"   Saved to: {save_path}")
    
    if args.train:
        print("\n  Starting post-conversion training...")
        # Re-run train with this as resume
        args.resume = save_path
        args.mode = None
        args.d_model = d_model
        args.vocab_size = vocab_size
        args.steps = args.steps or 500
        cmd_train(args)


def cmd_generate(args):
    """Generate text from a trained model."""
    if not args.model or not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    device = args.device or 'cpu'
    print(f"  Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device)
    cfg = checkpoint.get('config', RCVConfig())
    if isinstance(cfg, dict):
        cfg = RCVConfig(**cfg)
    
    model = NovaRCV(cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    prompt = args.prompt or "Once upon a time"
    tokens = args.tokens or 50
    temp = args.temperature or 0.7
    
    # Simple char tokenization for demo
    chars = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ,.-!?;:'0123456789\n"
    char_to_idx = {c: i % cfg.vocab_size for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    
    input_ids = torch.tensor([[char_to_idx.get(c, 0) for c in prompt[:32]]], device=device)
    
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=tokens, temperature=temp, top_p=0.9)
    
    text = ''.join(idx_to_char.get(i.item(), '?') for i in output[0])
    print(f"\n📝 Generated:\n{text}")


def cmd_test(args):
    """Run all tests."""
    print("  Running test suite...\n")
    from tests.test_rcv import run_all
    success = run_all()
    sys.exit(0 if success else 1)


def cmd_audit(args):
    """Run architecture audit."""
    print("  Running architecture audit...\n")
    from tools.audit import run_audit
    results = run_audit()
    print(f"\n  Audit complete: {results['issues']} issues found")
    sys.exit(0 if results['tests_passed'] else 1)


def cmd_fixes(args):
    """Verify all critical blockers are fixed."""
    print("  Verifying all fixes...\n")
    from tools.verify_fixes import main  # We'll just run the module
    exec(open(os.path.join(os.path.dirname(__file__), '..', 'tools', 'verify_fixes.py')).read())


def cmd_benchmark(args):
    """Run benchmarks."""
    print("  Running benchmarks...\n")
    from benchmarks.run_benchmarks import main
    sys.argv = ['run_benchmarks.py', '--dry-run'] if args.dry_run else ['run_benchmarks.py']
    main()


# ============================================================
# MAIN CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="NovaRCV - Complete Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m nova.cli train --model tiny
  python -m nova.cli train --d-model 1024 --vocab-size 50257 --steps 10000 --data corpus.txt
  python -m nova.cli convert --from gpt2 --save mymodel.pt
  python -m nova.cli generate --model checkpoints/nova_rcv_tiny_final.pt --prompt "Hello"
  python -m nova.cli test
  python -m nova.cli audit
        """
    )
    parser.add_argument('command', choices=['train', 'convert', 'generate', 'test', 'audit', 'benchmark'],
                       help='Command to run')
    
    # Global args
    parser.add_argument('--device', default=None, help='Device (cpu/cuda)')
    
    # Train args
    parser.add_argument('--model', default=None, help='Model config preset (tiny/small/medium/full_1b)')
    parser.add_argument('--mode', default=None, help='Alias for --model')
    parser.add_argument('--data', default=None, help='Training data file path')
    parser.add_argument('--steps', type=int, default=None, help='Training steps')
    parser.add_argument('--save', default=None, help='Save path for model')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    
    # Custom model params
    parser.add_argument('--d-model', type=int, default=None, help='Hidden dimension')
    parser.add_argument('--vocab-size', type=int, default=None, help='Vocabulary size')
    parser.add_argument('--slots', type=int, default=None, help='Number of memory slots')
    parser.add_argument('--iterations', type=int, default=None, help='Max iterations per token')
    parser.add_argument('--expansion', type=int, default=None, help='MLP expansion factor')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=None, help='Sequence length')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--min-iter', type=int, default=None, help='Min iterations')
    parser.add_argument('--max-iter-cap', type=int, default=None, help='Max iteration cap')
    
    # Convert args
    parser.add_argument('--from', dest='source', default=None, help='Source model name (HuggingFace)')
    parser.add_argument('--train', action='store_true', help='Train after conversion')
    
    # Generate args
    parser.add_argument('--prompt', default=None, help='Generation prompt')
    parser.add_argument('--tokens', type=int, default=50, help='Tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    
    # Benchmark args
    parser.add_argument('--dry-run', action='store_true', help='Dry run benchmarks')
    
    args = parser.parse_args()
    
    # Route commands
    if args.command == 'train':
        # Allow --model or --mode
        if not args.model and args.mode:
            args.model = args.mode
        cmd_train(args)
    elif args.command == 'convert':
        cmd_convert(args)
    elif args.command == 'generate':
        cmd_generate(args)
    elif args.command == 'test':
        cmd_test(args)
    elif args.command == 'audit':
        cmd_audit(args)
    elif args.command == 'benchmark':
        cmd_benchmark(args)


if __name__ == '__main__':
    main()