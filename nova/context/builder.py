from .scorer import ConceptScorer

from .selector import ContextSelector


class DynamicContextBuilder:

    def __init__(self):

        self.scorer = ConceptScorer()

        self.selector = ContextSelector()

    def build(

        self,

        query,

        graph,

    ):

        concepts = list(

            graph.nodes.values()

        )

        ranked = self.scorer.score(

            query,

            concepts,

        )

        selected = self.selector.select(

            ranked,

            limit=32,

        )

        return selected