from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any


class GoalType(Enum):
    UNDERSTAND = auto()
    RETRIEVE = auto()
    REASON = auto()
    VERIFY = auto()
    RESPOND = auto()


@dataclass(slots=True)
class Goal:

    goal_type: GoalType

    description: str

    priority: int = 1

    metadata: Dict[str, Any] = field(default_factory=dict)