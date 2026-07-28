from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any
import uuid
import time


@dataclass(slots=True)
class BrainState:

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    version: int = 0

    timestamp: float = field(default_factory=time.time)

    active_concepts: Dict[str, float] = field(default_factory=dict)

    working_memory: Dict[str, Any] = field(default_factory=dict)

    long_memory: Dict[str, Any] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def activate(
        self,
        concept: str,
        score: float,
    ):

        current = self.active_concepts.get(concept, 0.0)

        self.active_concepts[concept] = max(
            current,
            score,
        )

    def deactivate(
        self,
        threshold: float = 0.05,
    ):

        remove = []

        for k, v in self.active_concepts.items():

            if v < threshold:

                remove.append(k)

        for k in remove:

            del self.active_concepts[k]

    def decay(
        self,
        factor: float = 0.95,
    ):

        for k in list(self.active_concepts):

            self.active_concepts[k] *= factor

        self.deactivate()