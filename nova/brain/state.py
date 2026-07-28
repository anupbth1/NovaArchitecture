from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class BrainState:

    step: int = 0

    finished: bool = False

    data: Dict[str, Any] = field(default_factory=dict)

    result = None

    error = None