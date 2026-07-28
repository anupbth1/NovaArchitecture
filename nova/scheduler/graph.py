class ComputeGraph:

    def __init__(self):

        self.nodes = {}

    def add(self, node):

        self.nodes[node.id] = node

    def get(self, node_id):

        return self.nodes[node_id]