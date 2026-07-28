from __future__ import annotations

from collections import deque


class WavePropagationEngine:
    """
    Breadth-first activation propagation.

    Only visits reachable nodes.
    """

    def __init__(
        self,
        activation_decay: float = 0.70,
        min_activation: float = 0.05,
        max_depth: int = 8,
    ):
        self.activation_decay = activation_decay
        self.min_activation = min_activation
        self.max_depth = max_depth

    def propagate(
        self,
        store,
        start_id: str,
        initial_activation: float = 1.0,
    ):

        queue = deque()

        visited = {}

        queue.append(
            (
                start_id,
                initial_activation,
                0,
            )
        )

        while queue:

            node_id, score, depth = queue.popleft()

            if depth > self.max_depth:
                continue

            if score < self.min_activation:
                continue

            crystal = store.get(node_id)

            if crystal is None:
                continue

            if visited.get(node_id, -1) >= score:
                continue

            visited[node_id] = score

            crystal.activate(score)

            for edge in store.relations:

                if edge.source != node_id:
                    continue

                next_score = (
                    score
                    * edge.weight
                    * self.activation_decay
                )

                queue.append(
                    (
                        edge.target,
                        next_score,
                        depth + 1,
                    )
                )

        return visited