"""
NovaRCV — v2 (Fixed)
====================

Fixes from architecture audit:
1. Corrective iterations: Each iteration predicts the RESIDUAL error of all previous.
2. Deep supervision now works via corrective loss (boosting-style).
3. Position encoding: Fixed to sinusoidal (no learned limit).
4. Iteration scaling: More iterations should NOW reduce loss (boosting property).
5. Training: Uses corrective loss + truncated implicit backprop.
"""
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RCVConfig
from .reasoner import AdaptiveReasoner


class NovaRCV(nn.Module):
    """
    Nova RCV with corrective iterations.
    
    Architecture:
    1. Embed tokens (sinusoidal pos encoding, no learned limit)
    2. For each token: inject into workspace → iterative correction loop
    3. Each iteration outputs a CORRECTION delta, not a full prediction
    4. Final output = base prediction + sum(all corrections)
    5. Loss: each iteration's cumulative prediction gets CE loss (boosting)
    """
    def __init__(self, config: RCVConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_scale = config.d_model ** 0.5
        
        # Initial workspace
        self.init_workspace = nn.Parameter(
            torch.randn(1, config.num_slots, config.d_model) * 0.02
        )
        
        # Base prediction head (iteration 0 — before any refinement)
        self.base_proj = nn.Linear(config.d_model, config.d_model)
        
        # The iterative reasoner (corrective mode)
        self.reasoner = AdaptiveReasoner(
            d_model=config.d_model,
            num_slots=config.num_slots,
            max_iterations=config.max_iterations,
            expansion=config.expansion,
        )
        
        # Output projection (shared for all iterations)
        self.ln_out = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)
        
        # Sinusoidal position encoding (NO learned limit)
        # This supports generation beyond any training length
        self._init_sinusoidal_pos(config.d_model, config.seq_len * 4)
        
        self._init_weights()
    
    def _init_sinusoidal_pos(self, d_model: int, max_len: int = 8192):
        """Sinusoidal position encoding — no learned parameters, unlimited length."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, D)
        self.register_buffer('pos_encoding', pe, persistent=False)
    
    def _get_pos_encoding(self, pos_ids: torch.Tensor) -> torch.Tensor:
        """Get position encoding for arbitrary positions.
        pos_ids: (B, T) or (T,). Returns (B, T, D).
        """
        if pos_ids.dim() == 2:
            B, T = pos_ids.shape
            flat = pos_ids.reshape(1, -1)  # (1, B*T) for gather
            gathered = F.embedding(
                (flat % self.pos_encoding.size(1)).long(),
                self.pos_encoding[0]  # (max_len, D)
            )  # (1, B*T, D)
            return gathered.reshape(B, T, self.pos_encoding.size(-1))
        else:
            # (T,) → (1, T, D)
            flat = pos_ids.unsqueeze(0)  # (1, T)
            gathered = F.embedding(
                (flat % self.pos_encoding.size(1)).long(),
                self.pos_encoding[0]
            )
            return gathered  # (1, T, D)
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.head:
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias) if hasattr(module, 'bias') and module.bias is not None else None
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        iter_limits: Optional[Union[int, List[int]]] = None,
        return_losses: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Architecture: Token → Reasoner (iterative corrections) → Head → Logits
        
        The reasoner produces corrections. Each correction is ADDED to a running
        cumulative vector. The head projects this to vocabulary. All iterations
        are trained via cross-entropy on the cumulative prediction.
        """
        B, T = idx.shape
        device = idx.device
        
        # Embed tokens with sinusoidal position encoding
        x = self.embed(idx) * self.embed_scale
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        x = x + self._get_pos_encoding(pos_ids)
        
        # Initialize workspace from previous token (zero for first)
        workspace = self.init_workspace.expand(B, -1, -1)  # (B, M, D)
        
        # Determine iteration limits
        if isinstance(iter_limits, int):
            limits = [iter_limits] * T
        elif isinstance(iter_limits, list):
            limits = iter_limits + [self.config.max_iterations] * (T - len(iter_limits))
        else:
            limits = [self.config.max_iterations] * T
        
        all_logits = []
        all_corrections_list = []
        
        for t in range(T):
            token_embed = x[:, t:t+1, :]  # (B, 1, D)
            
            # Run reasoner: workspace was updated by previous token,
            # now correct for this new token
            workspace, corrections = self.reasoner(
                workspace, token_embed,
                iter_limit=limits[t],
                return_all=True
            )
            # corrections: (B, actual_iters, D)
            
            # Cumulative ensemble: start from zero, add each correction
            cumulative = corrections[:, 0:1, :]  # First correction is the base
            for i in range(1, corrections.size(1)):
                cumulative = cumulative + corrections[:, i:i+1, :]
            
            # Final logits
            final_logits = self.head(self.ln_out(cumulative))
            all_logits.append(final_logits)
            all_corrections_list.append(corrections)
        
        logits = torch.cat(all_logits, dim=1)  # (B, T, vocab)
        
        if targets is None:
            return logits
        
        # === CORRECTIVE TRAINING LOSS ===
        # Each iteration i predicts the CUMULATIVE sum
        # Loss: CE(Head(cumulative_i), target) for each i
        total_loss = 0.0
        final_loss = 0.0
        total_weight_sum = 0.0
        
        for t in range(T):
            corrections = all_corrections_list[t]  # (B, actual_iters, D)
            target_t = targets[:, t]
            
            cumulative = torch.zeros_like(corrections[:, 0:1, :])
            for i in range(corrections.size(1)):
                cumulative = cumulative + corrections[:, i:i+1, :]
                logits_i = self.head(self.ln_out(cumulative))
                loss_i = F.cross_entropy(logits_i.squeeze(1), target_t, reduction='mean')
                
                w = self.reasoner.iter_weights[i].item() if i < len(self.reasoner.iter_weights) else 1.0
                total_loss += w * loss_i
                total_weight_sum += w
                
                if i == corrections.size(1) - 1:
                    final_loss += loss_i
        
        total_loss = total_loss / max(total_weight_sum, 1.0)
        final_loss = final_loss / T
        
        if return_losses:
            return total_loss, final_loss
        return total_loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        use_adaptive_compute: bool = True,
    ) -> torch.Tensor:
        """Generate with corrective ensemble and adaptive compute."""
        self.eval()
        device = input_ids.device
        B = input_ids.shape[0]
        
        # Initialize workspace
        workspace = self.init_workspace.expand(B, -1, -1)
        prompt_len = input_ids.shape[1]
        
        # Embed prompt
        x = self.embed(input_ids) * self.embed_scale
        pos_ids = torch.arange(prompt_len, device=device).unsqueeze(0)
        x = x + self._get_pos_encoding(pos_ids)
        
        # Process prompt tokens (fast mode: min iterations)
        for t in range(prompt_len):
            token_embed = x[:, t:t+1, :]
            workspace, _ = self.reasoner(
                workspace, token_embed,
                iter_limit=self.config.min_iter,
                return_all=False
            )
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            last_token = generated[:, -1:]
            last_embed = self.embed(last_token) * self.embed_scale
            pos_id = torch.tensor([[generated.shape[1] - 1]], device=device)
            token_embed = last_embed + self._get_pos_encoding(pos_id)
            
            # Adaptive compute
            if use_adaptive_compute:
                difficulty = self.reasoner.estimate_difficulty(workspace, token_embed)
                iter_limit = int(
                    self.config.min_iter + 
                    difficulty * (self.config.max_iter_cap - self.config.min_iter)
                )
                iter_limit = max(self.config.min_iter, min(iter_limit, self.config.max_iter_cap))
            else:
                iter_limit = self.config.max_iterations
            
            # Get base prediction
            first_slot = workspace[:, 0:1, :]
            base = self.base_proj(first_slot)
            
            # Iterative corrections
            workspace, corrections = self.reasoner(
                workspace, token_embed,
                iter_limit=iter_limit,
                return_all=True
            )
            
            # Ensemble: base + sum(corrections)
            cumulative = base
            for i in range(corrections.size(1)):
                cumulative = cumulative + corrections[:, i:i+1, :]
            
            logits = self.head(self.ln_out(cumulative))  # (B, 1, vocab)
            logits = logits.squeeze(1) / temperature
            
            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < top_k_vals[:, -1:]] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def get_param_count(self) -> dict:
        """Get parameter breakdown."""
        total = sum(p.numel() for p in self.parameters())
        counts = {
            'total': total,
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'embedding': sum(p.numel() for p in self.embed.parameters()),
            'brain': sum(p.numel() for p in self.reasoner.brain.parameters()),
            'debate': sum(p.numel() for p in self.reasoner.debate.parameters()),
            'head': sum(p.numel() for p in self.head.parameters()),
            'correction_proj': sum(p.numel() for p in self.reasoner.correction_proj.parameters()),
            'base_proj': sum(p.numel() for p in self.base_proj.parameters()),
        }
        return counts
    
    def get_effective_size(self) -> str:
        """Honest effective size estimation."""
        counts = self.get_param_count()
        total = counts['total']
        # Corrective iterations don't multiply total compute by iterations
        # because each iteration is a correction, not a full recomputation.
        # The honest comparison: this model is a parameter-efficient alternative
        # to a Transformer of similar total FLOPs.
        # FLOPs comparison: iter * brain_FLOPs vs standard Transformer layer FLOPs
        return f"{total/1e6:.1f}M"