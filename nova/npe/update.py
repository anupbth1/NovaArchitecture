class PatternUpdater:

    def update(

        self,

        state,

        pid,

        reward,

    ):

        old = state.patterns.get(

            pid,

            0.0,

        )

        state.patterns[pid] = (

            old*0.99

            +

            reward*0.01

        )

        state.age[pid] = 0