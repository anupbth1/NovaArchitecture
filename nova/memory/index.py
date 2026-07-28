from collections import defaultdict


class MemoryIndex:

    def __init__(self):

        self.index = defaultdict(set)

    def add(self, value, uid):

        self.index[value.lower()].add(uid)

    def search(self, value):

        return list(

            self.index.get(value.lower(), [])

        )