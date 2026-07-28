from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import time
import uuid


@dataclass(slots=True)
class Crystal:

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    name: str = ""

    kind: str = ""

    confidence: float = 1.0

    activation: float = 0.0

    created_at: float = field(default_factory=time.time)

    last_used: float = field(default_factory=time.time)

    metadata: Dict = field(default_factory=dict)

    def activate(self, amount: float):

        self.activation = min(
            1.0,
            self.activation + amount,
        )

        self.last_used = time.time()

    def decay(self, factor=0.95):

        self.activation *= factor