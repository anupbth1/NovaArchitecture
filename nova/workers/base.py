from __future__ import annotations

from abc import ABC, abstractmethod


class Worker(ABC):
    """
    Base class for every Nova worker.

    Workers perform one specialized task.
    """

    name = "worker"

    priority = 0

    @abstractmethod
    def can_handle(
        self,
        state,
    ) -> bool:
        ...

    @abstractmethod
    def execute(
        self,
        state,
        context,
    ):
        ...