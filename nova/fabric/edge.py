from dataclasses import dataclass


@dataclass(slots=True)
class FabricEdge:

    source: str

    target: str

    weight: float = 1.0