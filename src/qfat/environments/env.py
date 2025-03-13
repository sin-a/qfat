from abc import ABC, abstractmethod
from typing import List, Union


class ConditionalEnv(ABC):
    @abstractmethod
    def sample_tasks(
        self,
        n_tasks: int,
        as_str: bool,
    ) -> Union[List[int], List[str]]:
        pass

    @abstractmethod
    def get_all_tasks(self, as_str: bool) -> Union[List[int], List[str]]:
        pass

    @abstractmethod
    def get_task_names(self, labels: List[int]) -> List[str]:
        pass
