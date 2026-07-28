class BoundaryDetector:

    WORD = {
        " ",
        "\t",
        "\n",
    }

    SENTENCE = {
        ".",
        "!",
        "?",
    }

    def feed(self, ch):

        if ch in self.WORD:
            return "word"

        if ch in self.SENTENCE:
            return "sentence"

        return None