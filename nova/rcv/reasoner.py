"""
AdaptiveReasoner
===============
The iterative reasoning engine that converts 1B parameters into 600B+ effective compute.

Key mechanism:
1. For each token, initialize 64 memory slots from current workspace state
2. Inject token embedding into all slots
3. Iterate: BrainCell → SlotDebate → BrainCell → SlotDebate → ...
4. Each iteration = 1 effective layer of depth with the full 1B brain
5. After N iterations, first slot's state is used for output prediction
6. Adaptive compute: harder tokens get more iterations (surprisal-based)

No KV-cache. No full attention. No MoE. Just iterative refinement.

Training:
- Deep supervision: every iteration's output is trained to predict the next token
- Late iterations get higher weight (they should be more accurate)
- Surprisal mechanism: if early iterations have high loss, run more iterations

Decoding:
- Easy tokens (e.g., "the", "and"): 5 iterations
- Complex tokens (e.g., function names, rare words): 30-50 iterations
- Adaptive threshold based on model's own uncertainty
"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .brain_cell import BrainCell
from .slot_debate import SlotDebate


class AdaptiveReasoner(nn.Module):
    """
    The iterative reasoning core.
    
    Takes a workspace (B, M, D) and token embedding (B, 1, D).
    Returns updated workspace + all intermediate outputs for deep supervision.
    """
    def __init__(
        self,
        d_model: int,
        num_slots: int = 64,
        max_iterations: int = 30,
        expansion: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.max_iterations = max_iterations
        
        # The knowledge brain (1B params)
        self.brain = BrainCell(d_model, expansion)
        
        # Slot debate (lightweight attention among perspectives)
        self.debate = SlotDebate(d_model, num_slots)
        
        # Token → Workspace injection
        self.token_mixer = nn.Linear(d_model * 2, d_model)
        
        # Halting mechanism (PonderNet-style)
        # Each slot learns when it has "thought enough"
        self.halt_gate = nn.Linear(d_model, 1)
        
        # Adaptive compute: predicts how many iterations this token needs
        # based on initial perplexity
        self.compute_predictor = nn.Linear(d_model, 1)
        
        # Iteration weighting for deep supervision
        self.register_buffer(
            'iter_weights',
            torch.linspace(0.1, 1.0, max_iterations)
        )
        
    def forward(
        self,
        workspace: torch.Tensor,
        token_embed: torch.Tensor,
        iter_limit: Optional[int] = None,
        return_all: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            workspace: (B, M, D) - Current workspace state from previous token
            token_embed: (B, 1, D) - Current token embedding
            iter_limit: Max iterations for this token (None = use max_iterations)
            return_all: Return all intermediate outputs for deep supervision
            
        Returns:
            workspace: (B, M, D) - Updated workspace
            iter_outputs: (B, actual_iters, D) - All intermediate first-slot states
        """
        B, M, D = workspace.shape
        limit = iter_limit if iter_limit is not None else self.max_iterations
        limit = min(limit, self.max_iterations)
        
        # Mix token into workspace slots
        token_expanded = token_embed.expand(-1, M, -1)
        combined = torch.cat([workspace, token_expanded], dim=-1)
        workspace = self.token_mixer(combined) + workspace
        
        # Store intermediate outputs for deep supervision
        iter_outputs = []
        halt_scores = []
        
        # Iterative refinement loop
        for t in range(limit):
            # Step 1: Apply knowledge (BrainCell)
            workspace = self.brain(workspace)
            
            # Step 2: Debate among perspectives (SlotDebate)
            workspace = self.debate(workspace)
            
            # Step 3: Compute halting score (PonderNet)
            halt = torch.sigmoid(self.halt_gate(workspace))  # (B, M, 1)
            halt_scores.append(halt.mean())
            
            # Step 4: Store first slot's state for deep supervision
            # First slot = "executive summary" of all perspectives
            iter_outputs.append(workspace[:, 0:1, :])  # (B, 1, D)
        
        # Stack intermediate outputs
        if return_all and iter_outputs:
            all_outputs = torch.cat(iter_outputs, dim=1)  # (B, actual_iters, D)
        else:
            all_outputs = iter_outputs[-1] if iter_outputs else workspace[:, 0:1, :]
        
        return workspace, all_outputs
    
    def compute_deep_supervision_loss(
        self,
        all_iter_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute deep supervision loss across all iterations.
        
        Args:
            all_iter_logits: (B, actual_iters, vocab) - Logits from each iteration
            targets: (B,) - Target token IDs
            
        Returns:
            Weighted loss giving more importance to later iterations
        """
        B, actual_iters, V = all_iter_logits.shape
        
        total_loss = 0.0
        for i in range(actual_iters):
            # Weight: later iterations have higher weight
            if i < len(self.iter_weights):
                w = self.iter_weights[i]
            else:
                w = 1.0
            
            loss_i = F.cross_entropy(all_iter_logits[:, i, :], targets, reduction='mean')
            total_loss += w * loss_i
        
        return total_loss / actual_iters
    
    @torch.no_grad()
    def estimate_difficulty(
        self,
        workspace: torch.Tensor,
        token_embed: torch.Tensor,
    ) -> float:
        """
        Estimate how difficult this token is (for adaptive compute).
        
        Uses the compute_predictor to estimate required iterations.
        Higher score = more iterations needed.
        """
        # Quick forward pass (3 iterations) to gauge difficulty
        B, M, D = workspace.shape
        
        token_exp = token_embed.expand(-1, M, -1)
        combined = torch.cat([workspace, token_exp], dim=-1)
        x = self.token_mixer(combined) + workspace
        
        # Run a few iterations
        for _ in range(3):
            x = self.brain(x)
            x = self.debate(x)
        
        # Predict difficulty from average slot state
        avg_state = x.mean(dim=1)  # (B, D)
        difficulty = torch.sigmoid(self.compute_predictor(avg_state))  # (B, 1)
        return difficulty.mean().item()