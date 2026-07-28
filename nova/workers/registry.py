class WorkerRegistry:

    def __init__(self):

        self._workers = []

    def register(self, worker):

        self._workers.append(worker)

        self._workers.sort(
            key=lambda x: x.priority,
            reverse=True,
        )

    def workers(self):

        return self._workers