from abc import ABC, abstractmethod


class NovaModule(ABC):

    @abstractmethod
    def forward(self, data):

        pass