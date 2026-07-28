from .state import EncoderState
from .event import StreamEvent, EventType
from .boundary import BoundaryDetector


class DirectSemanticEncoder:

    def __init__(self):

        self.state = EncoderState()

        self.boundary = BoundaryDetector()

    def feed(self, ch):

        self.state.position += 1

        b = self.boundary.feed(ch)

        if b is None:

            self.state.current_word += ch

            return None

        if b == "word":

            word = self.state.current_word

            self.state.current_word = ""

            if word:

                self.state.sentence.append(word)

                return StreamEvent(
                    EventType.TOKEN_END,
                    word,
                )

        if b == "sentence":

            if self.state.current_word:

                self.state.sentence.append(
                    self.state.current_word
                )

            sent = self.state.sentence

            self.state.current_word = ""
            self.state.sentence = []

            return StreamEvent(
                EventType.SENTENCE_END,
                sent,
            )

        return None