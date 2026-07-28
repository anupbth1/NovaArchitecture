class PromptBuilder:

    def build(

        self,

        graph,

        reasoning,

    ):

        prompt = []

        prompt.append(

            "You are Nova AI."

        )

        prompt.append("")

        prompt.append("Semantic Graph:")

        for node in graph.nodes.values():

            prompt.append(

                f"{node.semantic_type.name}: {node.value}"

            )

        prompt.append("")

        prompt.append("Reasoning Trace:")

        for step in reasoning:

            prompt.append(str(step))

        prompt.append("")

        prompt.append(

            "Generate the final answer."

        )

        return "\n".join(prompt)