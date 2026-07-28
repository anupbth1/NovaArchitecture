from dataclasses import dataclass, field
from typing import Callable


@dataclass(slots=True)
class ComputeNode:

    id: str

    cost: float

    priority: float

    execute: Callable

    dependencies: list[str] = field(default_factory=list)