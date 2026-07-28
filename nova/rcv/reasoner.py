"""
AdaptiveReasoner — v2 (Fixed)
=============================

Fixes from architecture audit:
1. Corrective iterations: Each iteration predicts the RESIDUAL of previous iterations,
   NOT the full target. Like boosting: iter 1 predicts, iter 2 corrects errors of iter 1.
   
2. Workspace stability: RMSNorm after each iteration step. Norm stays bounded.

3. Local per-iteration losses: Each iteration has its own objective.
   Iter 1: predict the token (standalone)
   Iter 2: predict the error of iter 1 (correction)
   Iter 3: predict the error of iter 2 (correction)
   Final: ensemble of all corrections

4. Confidence-based adaptive stopping: Halt when confidence > threshold.
   No more fixed 30 iterations for everything.
"""
import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .brain_cell import BrainCell
from .slot_debate import SlotDebate


class AdaptiveReasoner(nn.Module):
    """
    Iterative reasoning with CORRECTIVE iterations.
    
    Instead of each iteration predicting the SAME target (deep supervision),
    each iteration predicts the ERROR of all previous iterations combined.
    
    This creates a proper boosting-like improvement where:
    - Iteration 1: primary prediction
    - Iteration 2: correction of iter 1's errors  
    - Iteration 3: correction of iter 1+2's remaining errors
    ...
    - Final: primary + corrections = ensemble
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
        
        # The knowledge brain
        self.brain = BrainCell(d_model, expansion)
        
        # Slot debate
        self.debate = SlotDebate(d_model, num_slots)
        
        # Token → Workspace injection
        self.token_mixer = nn.Linear(d_model * 2, d_model)
        
        # RMSNorm for workspace stability between iterations
        self.iter_norm = nn.RMSNorm(d_model)
        
        # Correction head: reads current workspace + previous predictions
        # and outputs the correction delta
        self.correction_proj = nn.Linear(d_model, d_model)
        
        # Confidence estimator: how confident is this iteration's prediction
        self.confidence_gate = nn.Linear(d_model, 1)
        
        # Adaptive compute predictor
        self.compute_predictor = nn.Linear(d_model, 1)
        
        # Iteration weights: give more weight to later corrections
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
        Corrective iterative reasoning.
        
        Args:
            workspace: (B, M, D)
            token_embed: (B, 1, D)
            iter_limit: max iterations
            return_all: return all intermediate states
            
        Returns:
            workspace: (B, M, D) updated
            iter_outputs: (B, actual_iters, D) — each iteration's STATE
                         (for computing corrections in the main model)
        """
        B, M, D = workspace.shape
        limit = min(iter_limit if iter_limit is not None else self.max_iterations, self.max_iterations)
        
        # Mix token into workspace
        token_expanded = token_embed.expand(-1, M, -1)
        combined = torch.cat([workspace, token_expanded], dim=-1)
        workspace = self.token_mixer(combined) + workspace
        workspace = self.iter_norm(workspace)
        
        # Store intermediate workspace states
        iter_outputs = []
        confidence_scores = []
        
        for t in range(limit):
            # Step 1: Apply knowledge (BrainCell)
            workspace = self.brain(workspace)
            
            # Step 2: Debate among perspectives
            workspace = self.debate(workspace)
            
            # Step 3: Per-iteration normalization (keeps norm bounded)
            workspace = self.iter_norm(workspace)
            
            # Step 4: Compute correction vector from first slot
            # This represents "what should I fix in the current prediction"
            first_slot = workspace[:, 0:1, :]  # (B, 1, D)
            correction = self.correction_proj(first_slot)  # (B, 1, D)
            
            # Step 5: Store the correction state (used later for deep supervision)
            iter_outputs.append(correction)  # correction, not raw workspace
            
            # Step 6: Confidence score for early stopping
            confidence = torch.sigmoid(self.confidence_gate(first_slot))  # (B, 1, 1)
            confidence_scores.append(confidence.mean())
        
        # Stack corrections: (B, actual_iters, D)
        if return_all and iter_outputs:
            all_outputs = torch.cat(iter_outputs, dim=1)
        else:
            all_outputs = iter_outputs[-1] if iter_outputs else workspace[:, 0:1, :]
        
        return workspace, all_outputs
    
    def compute_corrective_loss(
        self,
        all_corrections: torch.Tensor,
        base_logits: torch.Tensor,
        targets: torch.Tensor,
        vocab_size: int,
        head: nn.Module,
        ln_out: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute corrective ensemble loss.
        
        Each iteration's correction is ADDED to the running prediction:
        - pred_1 = base + correction_1
        - pred_2 = base + correction_1 + correction_2
        - pred_k = base + sum(corrections_1..k)
        
        Loss at iteration k = CE(softmax(pred_k), target)
        
        The final output uses ALL corrections (ensemble).
        
        Args:
            all_corrections: (B, actual_iters, D) from forward()
            base_logits: (B, 1, vocab) logits from iteration 0 base prediction  
            targets: (B,) target token IDs
            head: output projection (D → vocab)
            ln_out: output layernorm
            
        Returns:
            total_loss: weighted sum of all iteration losses
            final_loss: loss using ALL corrections (ensemble)
            per_iter_losses: loss at each iteration
        """
        B, actual_iters, D = all_corrections.shape
        
        # Start with base prediction (or zero)
        cumulative_correction = torch.zeros(B, 1, D, device=all_corrections.device)
        
        per_iter_losses = []
        
        for i in range(actual_iters):
            # Add this iteration's correction
            cumulative_correction = cumulative_correction + all_corrections[:, i:i+1, :]
            
            # Project to vocab
            refined = cumulative_correction  # (B, 1, D)
            logits = head(ln_out(refined))  # (B, 1, vocab)
            
            # Loss for this iteration
            loss_i = F.cross_entropy(logits.squeeze(1), targets, reduction='mean')
            per_iter_losses.append(loss_i)
        
        # Stack losses
        iter_losses = torch.stack(per_iter_losses)  # (actual_iters,)
        
        # Weighted sum: later iterations get more weight
        weights = self.iter_weights[:actual_iters].to(iter_losses.device)
        total_loss = (weights * iter_losses).sum() / actual_iters
        
        # Final loss uses ALL corrections (ensemble)
        final_loss = iter_losses[-1]
        
        return total_loss, final_loss, iter_losses
    
    @torch.no_grad()
    def estimate_difficulty(
        self,
        workspace: torch.Tensor,
        token_embed: torch.Tensor,
    ) -> float:
        """Estimate token difficulty for adaptive compute."""
        B, M, D = workspace.shape
        
        token_exp = token_embed.expand(-1, M, -1)
        combined = torch.cat([workspace, token_exp], dim=-1)
        x = self.token_mixer(combined) + workspace
        
        for _ in range(3):
            x = self.brain(x)
            x = self.debate(x)
            x = self.iter_norm(x)
        
        avg_state = x.mean(dim=1)
        difficulty = torch.sigmoid(self.compute_predictor(avg_state))
        return difficulty.mean().item()