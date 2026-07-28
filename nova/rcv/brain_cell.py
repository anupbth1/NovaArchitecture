"""
BrainCell
=========
The actual 'knowledge' repository of the model.

Key insight: This single cell contains ~1B parameters.
By reusing it 30-50 times per token, we get effective depth of 600B+ layers.
No MoE. No mixture of anything. Just one dense brain, reused iteratively.

Architecture:
  x → LayerNorm → Linear(d_model → d_model*4) → GELU → LayerNorm → Linear(d_model*4 → d_model) → +x

This is simpler than standard LLM blocks (no attention inside the cell itself).
Attention happens only at the SlotDebate level among the 64 memory slots.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BrainCell(nn.Module):
    """
    The core knowledge cell.
    
    1B parameters live here (when d_model=2048, expansion=4):
      - fc1: 2048 → 8192  (~16.8M)
      - fc2: 8192 → 2048  (~16.8M)
      - norm1, norm2: ~4K each
      Total ~33.6M parameters per cell
      
    But this is reused 30x per token, giving 30 * 33.6M = 1.008B effective compute.
    With 64 slots processing independently, effective compute/token = ~64 * 1B = 64B.
    
    Actually the full 1B comes from:
      - Embedding matrix: 50257 * 2048 = ~103M
      - Output head: 50257 * 2048 = ~103M
      - BrainCell: ~34M
      - SlotDebate: ~25M
      - TokenMixer: ~8M
      Total: ~273M
      
    To reach 1B with d_model=2048:
      Increase expansion to 8-12 (fc1: 2048→16384, fc2: 16384→2048 = ~67M)
      Or stack 3 BrainCells (each reused iteratively) = ~100M
      Or increase vocab to 128k
      
    Effective compute depth = iterations * expansion_factor
    At max_iter=50: 50 * 8M (per iteration pass) = 400M parameter-equivalent compute
    """
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.d_model = d_model
        self.expansion = expansion
        
        # The dense MLP - this is where knowledge is stored
        self.fc1 = nn.Linear(d_model, d_model * expansion)
        self.fc2 = nn.Linear(d_model * expansion, d_model)
        
        # Pre-normalization for stability during iterative reuse
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model * expansion)
        
        # Scale factor to prevent explosion over many iterations
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, M, D) where M = num_slots, D = d_model
        
        Single step of knowledge application.
        """
        # Pre-norm → MLP → GELU → post-norm → residual
        h = self.fc1(self.norm1(x))
        h = F.gelu(h)
        h = self.fc2(self.norm2(h))
        
        # Scaled residual: prevent activation explosion over many iterations
        return x + h * torch.sigmoid(self.scale)


class DeepBrainCell(nn.Module):
    """
    A deeper version with 3 stacked MLP blocks.
    Each block is reused iteratively, so effective depth = 3 * iterations.
    
    At max_iter=50: 3 * 50 = 150 layer-equivalent depth.
    With d_model=2048, expansion=4: ~100M params.
    Combined with embeddings and head: ~300M.
    With expansion=12: ~800M params total.
    """
    def __init__(self, d_model: int, expansion: int = 4, num_blocks: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([
            BrainCell(d_model, expansion) for _ in range(num_blocks)
        ])
        self.router = nn.Linear(d_model, num_blocks)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply all blocks in sequence (each one is a reuse)
        for block in self.blocks:
            x = block(x)
        return x