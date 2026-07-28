class Planner:

    def build(

        self,

        graph,

        requested_nodes,

    ):

        plan = []

        visited = set()

        def dfs(node_id):

            if node_id in visited:

                return

            visited.add(node_id)

            node = graph.get(node_id)

            for dep in node.dependencies:

                dfs(dep)

            plan.append(node)

        for n in requested_nodes:

            dfs(n)

        return sorted(

            plan,

            key=lambda x: x.priority,

            reverse=True,
        )