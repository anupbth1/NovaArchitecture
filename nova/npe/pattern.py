from dataclasses import dataclass


@dataclass(slots=True)
class Pattern:

    id: int

    strength: float

    frequency: int

    embedding: list[float]