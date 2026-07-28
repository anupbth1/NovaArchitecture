"""
RCV Trainer
===========
Training pipeline for NovaRCV with:
- Deep supervision
- Adaptive compute (surprisal-based)
- Cosine LR schedule
- Gradient clipping
- Sample generation during training
"""
import os
import math
import time
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class RCVTrainer:
    """
    Trainer for NovaRCV model.
    
    Features:
    - Deep supervision with iteration weighting
    - Surprisal-based adaptive compute allocation
    - Per-token iteration limits
    - Cosine LR with warmup
    - Gradient clipping
    - Periodic text generation
    """
    def __init__(
        self,
        model: nn.Module,
        config,
        device: str = 'cuda',
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Optimizer (AdamW with weight decay)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )
        
        # Learning rate scheduler (cosine with warmup)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.total_steps,
            eta_min=config.lr * 0.1,
        )
        
        # Track training state
        self.step = 0
        self.running_surprisal = 1.0
        self.best_loss = float('inf')
        
    def _get_adaptive_iter_limits(
        self,
        batch_surprisal: float,
    ) -> int:
        """
        Determine iteration limit for next batch based on running surprisal.
        
        High surprisal = model is uncertain = more compute needed
        """
        cfg = self.config
        
        if self.running_surprisal > 2.0:
            # Very uncertain: use max iterations
            return cfg.max_iter_cap
        elif self.running_surprisal > 1.5:
            # Somewhat uncertain: increase iterations
            return cfg.max_iterations + 10
        elif self.running_surprisal < 0.8:
            # Very confident: decrease iterations
            return cfg.min_iter
        elif self.running_surprisal < 1.0:
            # Confident: slight decrease
            return max(cfg.min_iter, cfg.max_iterations - 5)
        else:
            # Default
            return cfg.max_iterations
    
    def train_step(self, input_ids: torch.Tensor) -> dict:
        """
        Single training step.
        
        Args:
            input_ids: (B, T) token IDs (labels are shifted inside)
            
        Returns:
            dict with loss values and stats
        """
        B, T = input_ids.shape
        device = input_ids.device
        
        # Shift for next-token prediction
        targets = input_ids[:, 1:].contiguous()  # (B, T-1)
        inputs = input_ids[:, :-1].contiguous()  # (B, T-1)
        
        # Determine adaptive iteration limit for this batch
        iter_limit = self._get_adaptive_iter_limits(self.running_surprisal)
        
        # Forward pass with deep supervision
        total_loss, per_token_losses = self.model(
            inputs,
            targets=targets,
            iter_limits=iter_limit,
            return_losses=True,
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping (critical for iterative models)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update running surprisal (EMA)
        avg_loss = per_token_losses.mean().item()
        self.running_surprisal = 0.9 * self.running_surprisal + 0.1 * avg_loss
        
        self.step += 1
        
        return {
            'loss': total_loss.item(),
            'surprisal': avg_loss,
            'running_surprisal': self.running_surprisal,
            'iter_limit': iter_limit,
            'lr': self.scheduler.get_last_lr()[0],
            'step': self.step,
        }
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, num_batches: int = 20) -> dict:
        """Evaluate model on validation data."""
        self.model.eval()
        total_loss = 0.0
        num_tokens = 0
        
        for i, batch in enumerate(val_loader):
            if i >= num_batches:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            
            loss = self.model(
                inputs,
                targets=targets,
                iter_limits=self.config.max_iterations,
            )
            
            total_loss += loss.item() * inputs.numel()
            num_tokens += inputs.numel()
        
        avg_loss = total_loss / num_tokens
        perplexity = math.exp(avg_loss)
        
        self.model.train()
        
        return {
            'val_loss': avg_loss,
            'val_perplexity': perplexity,
        }
    
    @torch.no_grad()
    def generate_sample(
        self,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 50,
    ) -> str:
        """Generate text sample for monitoring."""
        self.model.eval()
        
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )
        
        output = tokenizer.decode(generated[0], skip_special_tokens=True)
        self.model.train()
        return output
    
    def save_checkpoint(self, path: str, tokenizer=None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'step': self.step,
            'running_surprisal': self.running_surprisal,
            'best_loss': self.best_loss,
        }
        torch.save(checkpoint, path)
        print(f"✅ Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.running_surprisal = checkpoint['running_surprisal']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"✅ Checkpoint loaded from {path} (step {self.step})")
        return self