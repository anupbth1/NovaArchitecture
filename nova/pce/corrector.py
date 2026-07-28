class Corrector:

    def update(
        self,
        state,
        error,
        lr=0.10,
    ):
        return state + lr * error