from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(slots=True)
class Hypothesis:

    id: str

    statement: str

    confidence: float = 0.0

    evidence: List[str] = field(default_factory=list)

    missing: List[str] = field(default_factory=list)

    verified: bool = False