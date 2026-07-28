from dataclasses import dataclass


@dataclass(slots=True)
class Relation:

    source: str

    target: str

    relation: str

    weight: float = 1.0