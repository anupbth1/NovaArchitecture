from collections import defaultdict


class CognitiveFabric:

    def __init__(self):

        self.nodes = {}

        self.edges = defaultdict(list)

    def add_node(self, node):

        self.nodes[node.id] = node

    def connect(self, edge):

        self.edges[edge.source].append(edge)

    def neighbors(self, node_id):

        return self.edges[node_id]

    def activate(

        self,

        node_id,

        score,

    ):

        if node_id in self.nodes:

            self.nodes[node_id].activation += score