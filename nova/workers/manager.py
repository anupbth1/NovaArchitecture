from .registry import WorkerRegistry


class WorkerManager:

    def __init__(self):

        self.registry = WorkerRegistry()

    def register(self, worker):

        self.registry.register(worker)

    def execute(
        self,
        state,
        context,
    ):

        for worker in self.registry.workers():

            if worker.can_handle(state):

                state = worker.execute(
                    state,
                    context,
                )

        return state