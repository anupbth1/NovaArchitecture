"""
Concept Graph

Nova does NOT think using hidden vectors.

Nova thinks using connected semantic units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .semantic import SemanticUnit


@dataclass(slots=True)
class Relation:

    source: str

    target: str

    relation: str

    weight: float = 1.0


class ConceptGraph:

    def __init__(self):

        self.nodes: Dict[str, SemanticUnit] = {}

        self.edges: List[Relation] = []

    # -------------------------

    def add_node(self, node: SemanticUnit):

        self.nodes[node.uid] = node

    # -------------------------

    def connect(

        self,

        source: SemanticUnit,

        target: SemanticUnit,

        relation: str,

        weight: float = 1.0,

    ):

        self.edges.append(

            Relation(

                source.uid,

                target.uid,

                relation,

                weight,

            )

        )

    # -------------------------

    def neighbours(self, uid: str):

        out = []

        for edge in self.edges:

            if edge.source == uid:

                out.append(

                    self.nodes[edge.target]

                )

        return out

    # -------------------------

    def __len__(self):

        return len(self.nodes)

    def __repr__(self):

        return (

            f"ConceptGraph("

            f"nodes={len(self.nodes)}, "

            f"edges={len(self.edges)})"

        )