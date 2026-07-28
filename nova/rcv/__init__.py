"""
RCV (Recursive Computation Volume) Core
========================================
The heart of NovaArchitecture.

Key insight:
- 1 BrainCell with 1B parameters, reused iteratively
- Each iteration = +1 effective layer depth
- 30-50 iterations per token = 600B+ effective model depth
- No MoE needed: single dense brain handles all tasks
- Slot debate (64 slots) provides multi-perspective reasoning
- Adaptive compute: harder tokens get more iterations
"""
from .brain_cell import BrainCell
from .slot_debate import SlotDebate
from .reasoner import AdaptiveReasoner
from .nova_brain import NovaRCV
from .config import RCVConfig