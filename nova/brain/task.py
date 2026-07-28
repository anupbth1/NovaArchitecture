from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict
import uuid


class TaskType(Enum):
    UNDERSTAND = auto()
    RETRIEVE = auto()
    REASON = auto()
    PLAN = auto()
    VERIFY = auto()
    GENERATE = auto()


@dataclass(slots=True)
class BrainTask:

    id: str
    task_type: TaskType
    payload: Dict[str, Any]
    priority: int = 0
    completed: bool = False

    @staticmethod
    def create(task_type: TaskType, **payload):

        return BrainTask(
            id=str(uuid.uuid4()),
            task_type=task_type,
            payload=payload,
        )