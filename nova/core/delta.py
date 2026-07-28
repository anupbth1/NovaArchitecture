from dataclasses import dataclass
from typing import Dict


@dataclass(slots=True)
class StateDelta:

    activate: Dict[str, float]

    deactivate: list[str]