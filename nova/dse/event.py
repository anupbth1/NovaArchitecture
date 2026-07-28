from dataclasses import dataclass
from enum import Enum, auto


class EventType(Enum):
    CHAR = auto()
    TOKEN_END = auto()
    SENTENCE_END = auto()


@dataclass(slots=True)
class StreamEvent:
    event: EventType
    value: str