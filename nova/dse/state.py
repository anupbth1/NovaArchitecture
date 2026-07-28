from dataclasses import dataclass, field


@dataclass
class EncoderState:

    buffer: str = ""

    current_word: str = ""

    sentence: list[str] = field(default_factory=list)

    position: int = 0