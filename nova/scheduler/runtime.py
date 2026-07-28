class Runtime:

    def execute(

        self,

        plan,

        state,

    ):

        outputs = {}

        for node in plan:

            outputs[node.id] = node.execute(

                state,

                outputs,

            )

        return outputs