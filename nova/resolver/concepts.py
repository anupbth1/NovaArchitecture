from dataclasses import dataclass
from enum import Enum, auto

class ConceptType(Enum):

    UNKNOWN=auto()

    ENTITY=auto()

    ACTION=auto()

    OBJECT=auto()

    ATTRIBUTE=auto()

    LOCATION=auto()

    TIME=auto()

    NUMBER=auto()

@dataclass(slots=True)
class Concept:

    text:str

    ctype:ConceptType

    confidence:float