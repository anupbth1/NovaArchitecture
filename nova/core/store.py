from .crystal import Crystal
from .relation import Relation
from collections import defaultdict


class Store:
    """
    CrystalStore for Nova's semantic graph memory.
    Stores Crystal nodes and their Relations (edges).
    """

    def __init__(self):
        self.crystals = {}
        self.forward = defaultdict(list)
        self.backward = defaultdict(list)

    def add(self, crystal):
        self.crystals[crystal.id] = crystal

    def connect(self, relation):
        self.forward[relation.source].append(relation)
        self.backward[relation.target].append(relation)

    def neighbors(self, crystal_id):
        return self.forward.get(crystal_id, [])

    def get(self, crystal_id):
        return self.crystals.get(crystal_id)

    def size(self):
        return len(self.crystals)