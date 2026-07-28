from dataclasses import dataclass, field
from typing import Dict


@dataclass(slots=True)
class PatternState:

    patterns: Dict[int, float] = field(default_factory=dict)

    age: Dict[int, int] = field(default_factory=dict)

    confidence: Dict[int, float] = field(default_factory=dict)