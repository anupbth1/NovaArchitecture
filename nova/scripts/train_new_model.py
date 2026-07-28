#!/usr/bin/env python3
"""
NovaRCV - Train a New LLM from Scratch
========================================
Run this on Colab, RunPod, or Local PC to train a new language model
using the RCV (corrective iteration) architecture.

Usage:
    # Train tiny model (CPU/Free Colab)
    python nova/scripts/train_new_model.py --mode tiny --steps 1000

    # Train small model (T4 GPU Colab)
    python nova/scripts/train_new_model.py --mode small --steps 5000 --device cuda

    # Train medium model (A100 RunPod)
    python nova/scripts/train_new_model.py --mode medium --steps 50000 --device cuda
    
    # Resume from checkpoint
    python nova/scripts/train_new_model.py --mode small --resume checkpoints/step_1000.pt
"""
import sys, os, math, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from nova.rcv.nova_brain import NovaRCV
from nova.rcv.config import RCVConfig


# ============================================================
# CONFIGURATIONS
# ============================================================

CONFIGS = {
    "tiny": RCVConfig(
        d_model=128, vocab_size=5000, num_slots=8, max_iterations=5,
        expansion=2, batch_size=4, seq_len=64, lr=3e-4, total_steps=1000,
        min_iter=2, max_iter_cap=10,
    ),
    "small": RCVConfig(
        d_model=512, vocab_size=32000, num_slots=32, max_iterations=10,
        expansion=4, batch_size=8, seq_len=128, lr=3e-4, total_steps=5000,
        min_iter=3, max_iter_cap=20,
    ),
    "medium": RCVConfig(
        d_model=1024, vocab_size=50257, num_slots=64, max_iterations=15,
        expansion=4, batch_size=4, seq_len=256, lr=2e-4, total_steps=50000,
        min_iter=3, max_iter_cap=25,
    ),
    "full_1B": RCVConfig(
        d_model=2048, vocab_size=50257, num_slots=64, max_iterations=30,
        expansion=4, batch_size=2, seq_len=512, lr=1.5e-4, total_steps=200000,
        min_iter=5, max_iter_cap=50,
    ),
}


# ============================================================
# DATA LOADING
# ============================================================

class TextDataset(torch.utils.data.Dataset):
    """Simple text dataset for training."""
    def __init__(self, text_path: str = None, seq_len: int = 128, vocab_size: int = 5000):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        if text_path is not None and os.path.exists(text_path):
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
            # Simple char-level tokenization for demo
            self.chars = sorted(list(set(text)))
            self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
            self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
            self.data = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)
            print(f"  Loaded {len(text)} chars, {len(self.chars)} unique chars")
        else:
            # Generate synthetic data for testing
            print(f"  No data found at {text_path}, using synthetic data")
            self.data = torch.randint(0, min(vocab_size, 100), (10000,))
            self.chars = list(range(min(vocab_size, 100)))
            
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        return self.data[idx:idx + self.seq_len]


def create_data_loader(config, text_path: str = None):
    """Create data loader from text file or synthetic data."""
    dataset = TextDataset(text_path, config.seq_len + 1, config.vocab_size)
    return torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, drop_last=True,
    )


# ============================================================
# TRAINING LOOP
# ============================================================

def train(config_name: str = "tiny", device: str = "cpu", 
          steps: int = None, resume: str = None, text_path: str = None):
    """Main training function."""
    
    config = CONFIGS[config_name]
    if steps is not None:
        config.total_steps = steps
    
    print("=" * 70)
    print(f"NovaRCV Training - Config: {config_name}")
    print("=" * 70)
    print(f"  d_model={config.d_model}, vocab={config.vocab_size}")
    print(f"  slots={config.num_slots}, iterations={config.max_iterations}")
    print(f"  batch_size={config.batch_size}, seq_len={config.seq_len}")
    print(f"  total_steps={config.total_steps}, device={device}")
    
    # Create model
    model = NovaRCV(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.total_steps, eta_min=config.lr * 0.1,
    )
    
    # Resume from checkpoint
    start_step = 0
    if resume and os.path.exists(resume):
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        print(f"  Resumed from step {start_step}")
    
    # Data
    loader = create_data_loader(config, text_path)
    
    # Training loop
    print(f"\n{'='*70}")
    print("Training...")
    print(f"{'='*70}")
    
    model.train()
    data_iter = iter(loader)
    start_time = time.time()
    
    for step in range(start_step, config.total_steps):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        
        x = batch.to(device)  # (B, T)
        targets = x[:, 1:].contiguous()  # Shift for next-token prediction
        inputs = x[:, :-1].contiguous()
        
        # Forward + loss
        loss = model(inputs, targets=targets, iter_limits=config.max_iterations)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Logging
        if step % 50 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = config.batch_size * config.seq_len * 50 / max(elapsed, 0.1)
            lr = scheduler.get_last_lr()[0]
            print(f"  Step {step:6d}/{config.total_steps} | "
                  f"Loss: {loss.item():.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Speed: {tokens_per_sec:.0f} tok/s")
            start_time = time.time()
        
        # Save checkpoint
        if step > 0 and step % 500 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'loss': loss.item(),
            }
            torch.save(checkpoint, f"checkpoints/step_{step}.pt")
            print(f"  💾 Saved checkpoint at step {step}")
        
        # Generate sample
        if step > 0 and step % 200 == 0:
            model.eval()
            with torch.no_grad():
                sample = generate_sample(model, config, device)
                print(f"  📝 Sample: {sample[:200]}")
            model.train()
    
    # Save final model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'step': config.total_steps,
        'model_state_dict': model.state_dict(),
        'config': config,
    }, f"checkpoints/nova_rcv_{config_name}_final.pt")
    print(f"\n✅ Training complete! Model saved to checkpoints/nova_rcv_{config_name}_final.pt")
    print(f"   Params: {total_params:,} | Config: {config_name}")
    
    return model


@torch.no_grad()
def generate_sample(model, config, device, prompt="Once upon a time"):
    """Generate a text sample for monitoring."""
    # Simple char-level generation for demo
    chars = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ,.\n!?"
    char_to_idx = {c: i % config.vocab_size for i, c in enumerate(chars)}
    
    input_ids = torch.tensor(
        [[char_to_idx.get(c, 0) for c in prompt[:16]]], 
        device=device
    )
    
    generated = model.generate(
        input_ids,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.9,
    )
    
    # Decode
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    text = ''.join(idx_to_char.get(i.item(), '?') for i in generated[0])
    return text


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train NovaRCV from scratch")
    parser.add_argument('--mode', default='tiny', 
                        choices=list(CONFIGS.keys()),
                        help='Model size configuration')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--steps', type=int, default=None, help='Override training steps')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint path')
    parser.add_argument('--data', default=None, help='Path to text file for training')
    
    args = parser.parse_args()
    
    train(
        config_name=args.mode,
        device=args.device,
        steps=args.steps,
        resume=args.resume,
        text_path=args.data,
    )