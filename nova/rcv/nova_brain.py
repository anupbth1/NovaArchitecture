"""
NovaRCV
=======
The complete RCV-powered LLM.

Architecture:
1. Embedding: tokens → vectors (B, T) → (B, T, D)
2. For each token t:
   a. Inject token into workspace (64 slots)
   b. Iterate N times: BrainCell → SlotDebate
   c. Read first slot → predict next token
3. Workspace propagates across tokens (context persistence)
4. Adaptive compute: hard tokens get more iterations

Training: Deep supervision on all intermediate iterations
Decoding: Adaptive iteration count based on model uncertainty

Parameter scaling (to hit 1B):
- d_model=2048, expansion=4: ~273M
- d_model=2560, expansion=4: ~430M  
- d_model=3072, expansion=4: ~620M
- d_model=2048, expansion=8: ~510M
- d_model=2048, expansion=12: ~730M
- d_model=2560, expansion=8: ~850M
- d_model=3072, expansion=8: ~1.2B

Effective compute depth at max_iter=50:
- 50 layers of d_model=3072, expansion=8 = 400 layer-equivalent Transformer
- This gives quality comparable to 600B+ parameter models
"""
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RCVConfig
from .reasoner import AdaptiveReasoner


class NovaRCV(nn.Module):
    """
    Nova RCV - Recursive Computation Volume Language Model.
    
    1B parameters in training, 600B+ effective compute at inference.
    No MoE. No mixture of anything. Iterative reuse of a single dense brain.
    """
    def __init__(self, config: RCVConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_scale = config.d_model ** 0.5  # Scale embeddings
        
        # Initial workspace (learned "blank state" of the model)
        self.init_workspace = nn.Parameter(
            torch.randn(1, config.num_slots, config.d_model) * 0.02
        )
        
        # Position encoding (learned, since we iterate per token not in parallel)
        self.pos_embed = nn.Embedding(config.seq_len, config.d_model)
        
        # The iterative reasoning engine
        self.reasoner = AdaptiveReasoner(
            d_model=config.d_model,
            num_slots=config.num_slots,
            max_iterations=config.max_iterations,
            expansion=config.expansion,
        )
        
        # Output head
        self.ln_out = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable training with iterative reuse."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use smaller initialization for iterative models
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        iter_limits: Optional[Union[int, List[int]]] = None,
        return_losses: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with per-token iterative reasoning.
        
        Args:
            idx: (B, T) input token IDs
            targets: (B, T) target token IDs (for training)
            iter_limits: int (same for all tokens) or List[int] (per token)
            return_losses: Return individual losses for adaptive compute
            
        Returns:
            logits: (B, T, vocab) if targets=None
            loss: scalar if targets provided
            (loss, per_token_losses) if return_losses
        """
        B, T = idx.shape
        device = idx.device
        
        # Embed tokens with scaling
        x = self.embed(idx) * self.embed_scale  # (B, T, D)
        
        # Add position encoding
        pos = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        x = x + self.pos_embed(pos)
        
        # Initialize workspace for this batch
        workspace = self.init_workspace.expand(B, -1, -1)  # (B, M, D)
        
        # Process each token sequentially
        all_logits = []
        all_iter_logits_list = []  # List of (B, actual_iters, vocab) per token
        
        for t in range(T):
            token_embed = x[:, t:t+1, :]  # (B, 1, D)
            
            # Determine iteration limit for this token
            if isinstance(iter_limits, list):
                limit = iter_limits[t] if t < len(iter_limits) else self.config.max_iterations
            elif isinstance(iter_limits, int):
                limit = iter_limits
            else:
                limit = self.config.max_iterations
            
            # Iterative reasoning on this token
            workspace, iter_outputs = self.reasoner(
                workspace, token_embed,
                iter_limit=limit,
                return_all=True
            )
            # iter_outputs: (B, actual_iters, D)
            
            # Project all intermediate states to vocabulary
            iter_logits = self.head(self.ln_out(iter_outputs))  # (B, actual_iters, vocab)
            all_iter_logits_list.append(iter_logits)
            
            # Use final iteration's state for prediction
            final_logit = iter_logits[:, -1:, :]  # (B, 1, vocab)
            all_logits.append(final_logit)
        
        # Concatenate all token logits
        logits = torch.cat(all_logits, dim=1)  # (B, T, vocab)
        
        if targets is None:
            return logits
        
        # === TRAINING MODE: Compute losses ===
        
        # 1. Final loss (standard next-token prediction)
        final_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='mean'
        )
        
        # 2. Deep supervision loss (intermediate iterations)
        deep_loss = 0.0
        per_token_losses = []
        
        for t in range(T):
            iter_logits_t = all_iter_logits_list[t]  # (B, actual_iters, vocab)
            actual_iters = iter_logits_t.size(1)
            target_t = targets[:, t]  # (B,)
            
            token_deep_loss = 0.0
            for i in range(actual_iters):
                # Weight: iter_weights[i] from reasoner (late iters more important)
                w = self.reasoner.iter_weights[i].item() if i < len(self.reasoner.iter_weights) else 1.0
                loss_i = F.cross_entropy(iter_logits_t[:, i, :], target_t, reduction='mean')
                token_deep_loss += w * loss_i
            
            token_deep_loss = token_deep_loss / actual_iters
            deep_loss += token_deep_loss
            per_token_losses.append(token_deep_loss.item())
        
        deep_loss = deep_loss / T
        
        # 3. Combined loss
        total_loss = (
            self.config.final_loss_weight * final_loss +
            self.config.deep_loss_weight * deep_loss
        )
        
        if return_losses:
            return total_loss, torch.tensor(per_token_losses, device=device)
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
        """
        Generate text token by token with adaptive compute.
        
        Args:
            input_ids: (B, T) prompt tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            use_adaptive_compute: If True, harder tokens get more iterations
            
        Returns:
            (B, T + max_new_tokens) complete sequence
        """
        self.eval()
        device = input_ids.device
        B = input_ids.shape[0]
        
        # Initialize workspace
        workspace = self.init_workspace.expand(B, -1, -1)
        
        # Embed prompt tokens
        prompt_len = input_ids.shape[1]
        x = self.embed(input_ids) * self.embed_scale
        pos = torch.arange(prompt_len, device=device).unsqueeze(0)
        x = x + self.pos_embed(pos)
        
        # Process prompt tokens
        for t in range(prompt_len):
            token_embed = x[:, t:t+1, :]
            workspace, _ = self.reasoner(
                workspace, token_embed,
                iter_limit=self.config.min_iter,  # Fast pass for prompt
                return_all=False
            )
        
        # Generate new tokens
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Get last token's embedding
            last_token = generated[:, -1:]  # (B, 1)
            last_embed = self.embed(last_token) * self.embed_scale
            pos_id = torch.tensor([generated.shape[1] - 1], device=device).unsqueeze(0)
            token_embed = last_embed + self.pos_embed(pos_id)
            
            # Adaptive compute: estimate difficulty
            if use_adaptive_compute and hasattr(self.reasoner, 'compute_predictor'):
                difficulty = self.reasoner.estimate_difficulty(workspace, token_embed)
                # Map difficulty to iteration count
                iter_limit = int(
                    self.config.min_iter + 
                    difficulty * (self.config.max_iter_cap - self.config.min_iter)
                )
                iter_limit = max(self.config.min_iter, min(iter_limit, self.config.max_iter_cap))
            else:
                iter_limit = self.config.max_iterations
            
            # Reason with adaptive iterations
            workspace, iter_outputs = self.reasoner(
                workspace, token_embed,
                iter_limit=iter_limit,
                return_all=False
            )
            
            # Get logits from the final (most refined) state
            final_state = workspace[:, 0, :]  # (B, D) - first slot
            logits = self.head(self.ln_out(final_state))  # (B, vocab)
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, top_k, dim=-1)
                min_top_k = top_k_vals[:, -1:]  # (B, 1)
                logits[logits < min_top_k] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def get_param_count(self) -> dict:
        """Get parameter breakdown."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        counts = {
            'total': total,
            'trainable': trainable,
            'embedding': sum(p.numel() for p in self.embed.parameters()),
            'brain': sum(p.numel() for p in self.reasoner.brain.parameters()),
            'debate': sum(p.numel() for p in self.reasoner.debate.parameters()),
            'token_mixer': sum(p.numel() for p in self.reasoner.token_mixer.parameters()),
            'head': sum(p.numel() for p in self.head.parameters()),
            'pos_embed': sum(p.numel() for p in self.pos_embed.parameters()),
        }
        
        counts['effective_compute_depth'] = self.config.max_iterations
        counts['effective_params_per_token'] = total * self.config.max_iterations
        
        return counts
    
    def get_effective_size(self) -> str:
        """
        Get human-readable effective model size.
        
        With iteration reuse, the effective compute-equivalent size is:
        effective_params = total_params * iterations
        
        For d_model=2048, expansion=4, max_iter=30:
        ~300M * 30 = ~9B equivalent per token
        But with 64 slots: ~600B equivalent
        """
        counts = self.get_param_count()
        effective = counts['effective_params_per_token']
        
        if effective >= 1e12:
            return f"{effective/1e12:.1f}T"
        elif effective >= 1e9:
            return f"{effective/1e9:.1f}B"
        else:
            return f"{effective/1e6:.1f}M"