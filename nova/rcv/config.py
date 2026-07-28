"""
RCV Configuration
=================
Scale parameters to hit 1B parameters total.
No MoE - single dense brain, iteratively reused.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RCVConfig:
    # ----- Model Architecture -----
    vocab_size: int = 50257           # GPT-2 tokenizer compatible
    d_model: int = 2048               # Hidden dimension (~1B params at expansion=4)
    num_slots: int = 64               # Memory slots for multi-perspective
    max_iterations: int = 30          # Base iterations per token
    expansion: int = 4                # MLP expansion factor (d_model * 4 = 8192)
    
    # ----- Adaptive Compute (Surprisal-based) -----
    min_iter: int = 5                 # Minimum iterations even for easy tokens
    max_iter_cap: int = 50            # Maximum iterations for very hard tokens
    adapt_sensitivity: float = 0.3    # How aggressively to adapt iterations
    
    # ----- Deep Supervision -----
    deep_supervision: bool = True     # Train intermediate iterations too
    final_loss_weight: float = 0.6    # Weight for final output loss
    deep_loss_weight: float = 0.4     # Weight for intermediate losses
    
    # ----- Training -----
    batch_size: int = 4               # Per GPU
    seq_len: int = 256                # Context length
    lr: float = 3e-4                  # Learning rate
    weight_decay: float = 0.1         # AdamW weight decay
    total_steps: int = 100000         # Total training steps
    warmup_steps: int = 1000          # LR warmup
    
    # ----- Generation -----
    temperature: float = 0.7
    top_p: float = 0.9
    
    def param_count(self) -> int:
        """Calculate expected parameter count"""
        # Embedding: vocab_size * d_model
        embed = self.vocab_size * self.d_model
        
        # BrainCell: d_model -> d_model*4 -> d_model (2 linear layers + 2 norms)
        brain = (self.d_model * self.d_model * self.expansion * 2) + (self.d_model * 2)
        
        # SlotDebate: QKV projection + output projection
        debate = (self.d_model * self.d_model * 3) + (self.d_model * self.d_model)
        
        # Token mixer: d_model*2 -> d_model
        mixer = self.d_model * 2 * self.d_model
        
        # Output head: d_model -> vocab
        head = self.d_model * self.vocab_size
        
        # Layer norms
        norms = self.d_model * 2  # ln_out + init_workspace is param
        
        total = embed + brain + debate + mixer + head + norms
        return total
    
    @property
    def effective_depth(self) -> int:
        """Effective depth after iterative reuse"""
        return self.max_iterations  # Each iteration = 1 layer of depth