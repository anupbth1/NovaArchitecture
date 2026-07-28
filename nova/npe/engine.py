class PatternEngine:

    def activate(

        self,

        state,

        pattern_ids,

    ):

        active = []

        for pid in pattern_ids:

            if pid not in state.patterns:

                continue

            score = state.patterns[pid]

            active.append(

                (

                    pid,

                    score,

                )

            )

        active.sort(

            key=lambda x:x[1],

            reverse=True,

        )

        return active