from __future__ import annotations

from dataclasses import dataclass
from typing import List

from nova.workers.base import Worker
from nova.representation.semantic import SemanticUnit


@dataclass(slots=True)
class MemoryHit:

    concept: SemanticUnit

    score: float


class MemoryWorker(Worker):

    name = "memory"

    priority = 90

    def can_handle(self, state):

        return "graph" in state.data

    def execute(self, state, context):

        graph = state.data["graph"]

        query = state.data.get("query", "")

        query_words = {

            x.lower()

            for x in query.split()

        }

        hits: List[MemoryHit] = []

        for node in graph.nodes.values():

            score = node.confidence

            if node.value.lower() in query_words:

                score += 10

            if score > 0:

                hits.append(

                    MemoryHit(

                        node,

                        score,

                    )

                )

        hits.sort(

            key=lambda x: x.score,

            reverse=True,

        )

        state.data["memory_hits"] = hits[:64]

        return state