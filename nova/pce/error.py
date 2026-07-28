import numpy as np

class ErrorComputer:

    def compute(
        self,
        predicted,
        observed,
    ):
        return observed - predicted

    def magnitude(
        self,
        error,
    ):
        return np.linalg.norm(error)