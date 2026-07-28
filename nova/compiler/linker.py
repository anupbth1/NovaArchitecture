from dataclasses import dataclass


@dataclass(slots=True)
class LinkInstruction:

    source: str

    relation: str

    target: str

    weight: float = 1.0