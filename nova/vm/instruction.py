from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class Opcode(Enum):

    LOAD = auto()

    STORE = auto()

    FIND = auto()

    FILTER = auto()

    COMPARE = auto()

    INFER = auto()

    VERIFY = auto()

    RETURN = auto()


@dataclass(slots=True)
class Instruction:

    opcode: Opcode

    operand: Any = None