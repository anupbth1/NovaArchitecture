class ContextSelector:

    def select(

        self,

        scored,

        limit=64,

    ):

        return [

            concept

            for _, concept

            in scored[:limit]

        ]