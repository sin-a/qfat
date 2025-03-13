from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from numpy.typing import NDArray

from qfat.datasets.dataset import Batch


@dataclass
class ModelOutput:
    output: Any
    loss: torch.Tensor


class InvalidBatchException(Exception):
    """Exception raised when an invalid batch is encountered.

    This exception is specifically raised when a batch is empty
    or does not meet the requirements for processing.

    Attributes:
        message (str): Explanation of the error.
    """

    def __init__(self, message="The batch is invalid or empty."):
        self.message = message
        super().__init__(self.message)


class GenerativeModel(nn.Module, metaclass=ABCMeta):
    @property
    @abstractmethod
    def is_conditional(self) -> bool:
        """An attribute determining if the model is conditional or not"""
        pass

    @abstractmethod
    def sample(
        self,
        x: Union[torch.Tensor, NDArray],
        return_output: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[ModelOutput], Dict[str, Any]]:
        """Samples from the generative model, given a context.

        Args:
            x (Union[torch.Tensor, NDArray]): The context to condition
                the sampling on.
            return_output (bool): Whether to return the model output or not.
        Returns:
            Tuple[torch.Tensor, Optional[ModelOutput], Dict[str, Any]]: A tuple of the sampled output,
                the model output and any sampling metadata associated with the sample. The first element is
                a 3D array of shape (batch_size, 1, output_dim), and the second element is the model output,
                which is None if return_output is set to False. The last element is a dict containing any
                important information about the sampling of the data point. If no metadata, return an empty dict.
        """
        pass

    @abstractmethod
    def forward(self, batch: Batch) -> ModelOutput:
        """The model forward function, supporting passing the targets y to compute a loss.

        Args:
           batch (Batch): Dataclass containing the batched x, y and a validiting mask
               denoting where the trajectory is valid.
        Returns:
            ModelOutput: A dataclass holding the output and the loss.
        """
        pass

    @abstractmethod
    def sample_from_model_output(
        self,
        x: ModelOutput,
        temperature: Optional[float] = None,
        last_only: bool = False,
    ) -> torch.Tensor:
        """Samples from the distribution given the model's output.

        Args:
            x (ModelOutput): Output from the model's forward pass.
            temperature (Optional[float], optional): Adjusts the uncertainty
                in the sampling. The higher, the higher the uncertainty.
                Defaults to None.
            last_only (bool, optional): Whether to sample only from the last
                output in the output sequence. Defaults to False.

        Returns:
            torch.Tensor: The sampled output.
        """
        pass
