from __future__ import annotations

from typing import List

from nova.representation.semantic import SemanticUnit


class ConceptScorer:
    """
    Scores semantic concepts for relevance.

    This is intentionally simple in v0.1.
    Future versions will use Nova Brain signals.
    """

    def score(
        self,
        query: str,
        concepts: List[SemanticUnit],
    ):

        query_words = {
            x.lower()
            for x in query.split()
        }

        scored = []

        for concept in concepts:

            score = 0.0

            value = concept.value.lower()

            if value in query_words:
                score += 10

            if any(
                w in value
                for w in query_words
            ):
                score += 5

            score += concept.confidence

            scored.append(
                (
                    score,
                    concept,
                )
            )

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        return scored