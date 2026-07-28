"""
NovaArchitecture - RCV Language Model
======================================

A next-generation LLM architecture based on Recursive Computation Volume (RCV).

Key Innovation: Instead of stacking layers (like Transformers),
RCV reuses a single "brain" iteratively per token. Each iteration
adds effective depth without adding parameters.

This gives: 1B parameter training cost with 600B+ effective compute.

Core Modules:
- rcv: The RCV implementation (BrainCell, SlotDebate, AdaptiveReasoner, NovaRCV)
- brain: Pipeline orchestration
- reasoning: Hypothesis generation and verification
- memory: Dynamic knowledge graph
- planner: Goal planning
- core: Low-level data structures
- representation: Semantic units and tokenization
- workers: Parallel execution
"""
from .rcv.nova_brain import NovaRCV
from .rcv.config import RCVConfig
from .rcv.brain_cell import BrainCell
from .rcv.slot_debate import SlotDebate
from .rcv.reasoner import AdaptiveReasoner
from .rcv.trainer import RCVTrainer

__version__ = "0.1.0"
__all__ = [
    'NovaRCV', 'RCVConfig', 'BrainCell', 'SlotDebate',
    'AdaptiveReasoner', 'RCVTrainer',
]