from dataclasses import dataclass
from enum import Enum, auto


class GoalType(Enum):

    ANSWER = auto()

    PLAN = auto()

    RETRIEVE = auto()

    VERIFY = auto()

    LEARN = auto()


@dataclass(slots=True)
class Goal:

    goal_type: GoalType

    text: str

    priority: int = 1