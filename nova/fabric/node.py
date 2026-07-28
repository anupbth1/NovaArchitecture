from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(slots=True)
class FabricNode:

    id: str

    kind: str

    value: Any

    activation: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)