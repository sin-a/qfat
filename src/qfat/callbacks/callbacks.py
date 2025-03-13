from abc import ABC, abstractmethod

from qfat.entrypoints.entrypoint import Entrypoint


class Callback(ABC):
    @abstractmethod
    def __call__(self, ep: Entrypoint) -> None:
        """Called at the allowed intervals"""
        pass

    def finalize(self) -> None:
        """Called at the end of the run"""
        return
