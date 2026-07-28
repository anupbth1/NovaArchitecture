class PCERuntime:

    def __init__(self):

        self.predictor = Predictor()

        self.error = ErrorComputer()

        self.corrector = Corrector()

    def step(
        self,
        state,
        observation,
    ):

        predicted = self.predictor.predict(state)

        err = self.error.compute(
            predicted,
            observation,
        )

        if self.error.magnitude(err) < 0.01:
            return state

        return self.corrector.update(
            state,
            err,
        )