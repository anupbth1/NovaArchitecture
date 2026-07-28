"""
Universal Semantic Representation (USR)

This module defines Nova's internal language.

Everything inside Nova works on Semantic Units,
NOT raw tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict
import uuid


# -------------------------------------------------------
# Semantic Types
# -------------------------------------------------------

class SemanticType(Enum):

    ENTITY = auto()

    ACTION = auto()

    OBJECT = auto()

    ATTRIBUTE = auto()

    LOCATION = auto()

    TIME = auto()

    NUMBER = auto()

    CONDITION = auto()

    RELATION = auto()

    UNKNOWN = auto()


# -------------------------------------------------------
# Semantic Unit
# -------------------------------------------------------

@dataclass(slots=True)
class SemanticUnit:

    uid: str

    semantic_type: SemanticType

    value: str

    confidence: float = 1.0

    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(
        semantic_type: SemanticType,
        value: str,
        confidence: float = 1.0,
        **metadata,
    ) -> "SemanticUnit":

        return SemanticUnit(

            uid=str(uuid.uuid4()),

            semantic_type=semantic_type,

            value=value,

            confidence=confidence,

            metadata=metadata,
        )

    def __repr__(self):

        return (
            f"<SemanticUnit "
            f"{self.semantic_type.name}: "
            f"{self.value}>"
        )