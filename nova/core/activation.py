class ActivationEngine:

    def spread(

        self,

        store,

        crystal,

        amount=1.0,

        decay=0.5,

    ):

        crystal.activate(amount)

        for edge in store.relations:

            if edge.source != crystal.id:

                continue

            target = store.get(edge.target)

            if target is None:

                continue

            target.activate(

                amount *

                edge.weight *

                decay

            )