from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class DeltaOp(Enum):

    ACTIVATE = auto()

    DEACTIVATE = auto()

    LINK = auto()

    UNLINK = auto()

    UPDATE = auto()


@dataclass(slots=True)
class DeltaInstruction:

    operation: DeltaOp

    target: str

    value: Any = None