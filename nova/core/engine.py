from .state import BrainState
from .delta import StateDelta


class PersistentBrain:

    def __init__(self):

        self.state = BrainState()

    def step(
        self,
        delta: StateDelta,
    ):

        self.state.version += 1

        for name, score in delta.activate.items():

            self.state.activate(
                name,
                score,
            )

        for name in delta.deactivate:

            self.state.active_concepts.pop(
                name,
                None,
            )

        self.state.decay()

        return self.state