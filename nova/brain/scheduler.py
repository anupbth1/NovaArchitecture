from collections import deque

class TaskScheduler:

    def __init__(self):

        self.queue = deque()

    def submit(self, task):

        self.queue.append(task)

    def next(self):

        if self.queue:
            return self.queue.popleft()

        return None

    def empty(self):

        return len(self.queue) == 0