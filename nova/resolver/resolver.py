from .concepts import *

from .matcher import *

class SemanticResolver:

    def __init__(self):

        self.matcher=ContextMatcher()

        self.memory={}

    def resolve(

        self,

        word,

        context,

    ):

        if word in self.memory:

            return self.memory[word]

        concept=Concept(

            text=word,

            ctype=ConceptType.UNKNOWN,

            confidence=0.0,

        )

        self.memory[word]=concept

        return concept