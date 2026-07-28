"""
BrainCell
=========
The core knowledge cell. Single dense MLP, iteratively reused.

Fix #1: Added output RMSNorm to prevent workspace activation explosion.
Previously: x + h * sigmoid(scale) — norm grew 0.49 → 14.70 over 10 iters.
Now: x + h * learned_gate, then RMSNorm — norm stays bounded.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BrainCell(nn.Module):
    """
    The core knowledge cell.
    
    Key change: Output is RMSNorm'd to prevent activation explosion over iterations.
    The scale gate controls how much new info to admit vs. keeping old info.
    """
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.d_model = d_model
        self.expansion = expansion
        
        # The dense MLP
        self.fc1 = nn.Linear(d_model, d_model * expansion)
        self.fc2 = nn.Linear(d_model * expansion, d_model)
        
        # Pre-normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model * expansion)
        
        # Learned gate: how much of the new info to admit (0 = none, 1 = all)
        # Initialized to admit 50% — prevents explosion while still allowing update
        self.learned_gate = nn.Parameter(torch.zeros(1))  # sigmoid(0) = 0.5
        
        # Output RMSNorm to keep workspace norm bounded
        self.out_norm = nn.RMSNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, M, D)
        Returns: (B, M, D) with stable norm
        """
        # Pre-norm → MLP → GELU → post-norm
        h = self.fc1(self.norm1(x))
        h = F.gelu(h)
        h = self.fc2(self.norm2(h))
        
        # Gated residual: workspace = x + gate * h
        # gate = sigmoid(learned_gate) ∈ (0, 1) — controls update strength
        gate = torch.sigmoid(self.learned_gate)
        out = x + gate * h
        
        # RMSNorm after iteration prevents norm growth
        # This is the KEY fix: without this, norm compounds over iterations
        out = self.out_norm(out)
        
        return out


class DeepBrainCell(nn.Module):
    """
    Deeper version with 3 stacked MLP blocks.
    Each with its own gate and output norm for stability.
    """
    def __init__(self, d_model: int, expansion: int = 4, num_blocks: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([
            BrainCell(d_model, expansion) for _ in range(num_blocks)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x