from dataclasses import dataclass

from nova.memory.memory import DynamicMemory


@dataclass
class BrainContext:

    memory: DynamicMemory

    max_steps: int = 32