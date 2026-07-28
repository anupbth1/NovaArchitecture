from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from nova.representation.semantic import SemanticUnit
from nova.representation.graph import Relation


@dataclass
class MemoryNode:

    concept: SemanticUnit

    relations: Set[str] = field(default_factory=set)


class DynamicMemory:

    def __init__(self):

        self.nodes: Dict[str, MemoryNode] = {}

        self.edges: List[Relation] = []

    # ---------------------

    def store(self, unit: SemanticUnit):

        if unit.uid not in self.nodes:

            self.nodes[unit.uid] = MemoryNode(unit)

    # ---------------------

    def connect(self, relation: Relation):

        self.edges.append(relation)

        if relation.source in self.nodes:

            self.nodes[relation.source].relations.add(relation.target)

    # ---------------------

    def get(self, uid: str):

        return self.nodes.get(uid)

    # ---------------------

    def neighbors(self, uid: str):

        if uid not in self.nodes:

            return []

        return [

            self.nodes[x]

            for x in self.nodes[uid].relations

            if x in self.nodes

        ]

    # ---------------------

    def size(self):

        return len(self.nodes)