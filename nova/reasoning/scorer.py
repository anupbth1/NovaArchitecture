class HypothesisScorer:

    def score(

        self,

        hypothesis,

    ):

        score = hypothesis.confidence

        score += len(hypothesis.evidence) * 0.2

        score -= len(hypothesis.missing) * 0.15

        return score