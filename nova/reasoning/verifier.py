class HypothesisVerifier:

    def verify(

        self,

        hypothesis,

    ):

        if hypothesis.confidence > 0.80:

            hypothesis.verified = True

        return hypothesis