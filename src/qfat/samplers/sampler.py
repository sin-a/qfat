import logging
from abc import ABC, abstractmethod
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from line_profiler import profile
from numpy.typing import NDArray

from qfat.models.generative_model import GenerativeModel, ModelOutput

logger = logging.getLogger(__name__)


class ModelSampler(ABC):
    """A convienience class to call its sample function without
    worying about any of the internals (e.g keeping track of observed
    history)
    """

    def __init__(self, model: GenerativeModel) -> None:
        super().__init__()
        self.model = model
        if model.training:
            logger.debug(
                "The passed model was in training mode. Setting it to eval mode."
            )
            self.model.eval()

    @abstractmethod
    def reset() -> None:
        pass

    @abstractmethod
    def sample(x: NDArray, **kwargs) -> NDArray:
        pass


class AutoRegressiveSampler(ModelSampler):
    def __init__(
        self,
        model: GenerativeModel,
        context_len: Optional[int] = None,
        temperature: float = 1,
        horizon: int = 0,
        use_tensors: bool = False,
        sample_fn: str = "modes",
    ) -> None:
        """Keeps track of inputs in a queue and passes it autoregressively to the model's sample function.

        Args:
            model (GenerativeModel): The generative model used for sampling.
            context_len (Optional[int], optional): The context length. Defaults to None, which
                will assume the model has an attribute 'context_len' which will be used.
            temperature (float, optional): A temperature to scale the variances of the model. Defaults to 1.
            use_tensors (bool, optional): If True, keeps the context in PyTorch tensors. Defaults to False (NumPy).
            sample_fn (str): Which sample function to set for the model. For QFAT, can be either "modes" or "gmm".
        Raises:
            ValueError: If the passed model has no attribute 'context_len'.
        """
        super().__init__(model)
        if hasattr(self.model, "context_len") and context_len is None:
            context_len = self.model.context_len
        if context_len is None:
            raise ValueError(
                "'context_len' was not passed and it was not found as an attribute of the model."
            )
        self.context = Queue(context_len)
        self.temperature = temperature
        self.horizon = horizon
        self.use_tensors = use_tensors
        self.model.sample_fn = sample_fn

    def _convert_to_context_format(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Converts input to the desired context format (NumPy or PyTorch).

        Args:
            x (Union[np.ndarray, torch.Tensor]): Input data.

        Returns:
            Union[np.ndarray, torch.Tensor]: Converted data.
        """
        if self.use_tensors:
            return x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
        else:
            return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

    def update_context(
        self, x: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Updates the context with the new observation.

        Args:
            x (Union[np.ndarray, torch.Tensor]): The new observation.

        Returns:
            Union[np.ndarray, torch.Tensor]: The updated context in the appropriate format.
        """
        if not isinstance(x, list):
            x = [x]
        for _x in x:
            _x = self._convert_to_context_format(_x)
            if self.context.full():
                self.context.get(timeout=0)
            self.context.put(_x, timeout=0)

        if self.use_tensors:
            return torch.stack(list(self.context.queue), dim=0)
        else:
            return np.stack(list(self.context.queue), axis=0)

    def reset(self) -> None:
        """Clears the context."""
        with self.context.mutex:
            self.context.queue.clear()

    @profile
    def sample(
        self,
        x: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]],
        model_kwargs: Optional[Dict] = None,
        return_output: bool = True,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[ModelOutput], Dict[str, Any]]:
        """Samples from the underlying model, while keeping track of the previously passed context.

        Args:
            x (Union[np.ndarray, torch.Tensor]): The new observation to add to the context.
            model_kwargs (Optional[Dict]): Keyword arguments for the model's sample function.
            return_output (bool): Whether to return the model output. Defaults to True.

        Returns:
            Tuple[Union[np.ndarray, torch.Tensor], Optional[ModelOutput], Dict[str, Any]]:
                A tuple of the sampled output, the model forward pass output, and the sample metadata.
        """
        if model_kwargs is None:
            model_kwargs = {}
        x = self.update_context(x=x)[None, ...]  # add batch dimension
        output, model_out, metadata = self.model.sample(
            x=x, temperature=self.temperature, **model_kwargs
        )

        return (
            output.squeeze((0, 1)).cpu().numpy(),
            model_out if return_output else None,
            metadata,
        )


class DualContextAutoRegressiveSampler(ModelSampler):
    def __init__(
        self,
        model: GenerativeModel,
        state_context_len: Optional[int] = None,
        action_context_len: Optional[int] = None,
        temperature: float = 1,
        use_tensors: bool = False,
    ) -> None:
        """
        Keeps track of both state and action contexts and passes them autoregressively to the model's sample function.
        The states and the actions are concatenated together in the dimension axis.

        Args:
            model (GenerativeModel): The generative model used for sampling.
            state_context_len (Optional[int]): The length of the state context.
            action_context_len (Optional[int]): The length of the action context.
            temperature (float): A temperature to scale the variances of the model.
            use_tensors (bool): If True, keeps the context in PyTorch tensors. Defaults to False (NumPy).
        """
        super().__init__(model)

        if hasattr(self.model, "context_len") and state_context_len is None:
            state_context_len = self.model.context_len
        if state_context_len is None:
            raise ValueError(
                "'state_context_len' was not passed and it was not found as an attribute of the model."
            )
        if action_context_len is None:
            action_context_len = state_context_len

        self.state_context = Queue(state_context_len)
        self.action_context = Queue(action_context_len)
        self.temperature = temperature
        self.use_tensors = use_tensors
        self.horizon = 0

        self._update_queue(
            self.action_context,
            torch.zeros(self.model.out_dim).to(self.model.device)
            if self.use_tensors
            else np.zeros(self.model.out_dim),
        )

    def _convert_to_context_format(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Converts input to the desired context format (NumPy or PyTorch).
        """
        if self.use_tensors:
            return x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
        else:
            return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

    def _update_queue(
        self, queue: Queue, item: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """
        Updates the given queue with a new item, maintaining its size limit.

        Args:
            queue (Queue): The queue to update.
            item (Union[np.ndarray, torch.Tensor]): The item to add to the queue.
        """
        if queue.full():
            queue.get(timeout=0)
        queue.put(item, timeout=0)

    def reset(self) -> None:
        """
        Clears the state and action contexts.
        """
        self.state_context.queue.clear()
        self.action_context.queue.clear()
        self._update_queue(
            self.action_context,
            torch.zeros(self.model.out_dim).to(self.model.device)
            if self.use_tensors
            else np.zeros(self.model.out_dim),
        )

    def update_contexts(
        self,
        state: Optional[Union[np.ndarray, torch.Tensor]] = None,
        action: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Updates the state and action contexts.

        Args:
            state (Optional[Union[np.ndarray, torch.Tensor]]): The new state observation.
            action (Optional[Union[np.ndarray, torch.Tensor]]): The new action observation.

        Returns:
            Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
                The updated state and action contexts.
        """
        if state is not None:
            state = self._convert_to_context_format(state)
            self._update_queue(self.state_context, state)

        if action is not None:
            action = self._convert_to_context_format(action)
            self._update_queue(self.action_context, action)

        if self.use_tensors:
            state_context = torch.stack(list(self.state_context.queue), dim=0)
            action_context = torch.stack(list(self.action_context.queue), dim=0)
        else:
            state_context = np.stack(list(self.state_context.queue), axis=0)
            action_context = np.stack(list(self.action_context.queue), axis=0)

        return state_context, action_context

    def sample(
        self,
        x: Union[np.ndarray, torch.Tensor],
        model_kwargs: Optional[Dict] = None,
        return_output: bool = True,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[ModelOutput], Dict[str, Any]]:
        """
        Samples from the underlying model, while keeping track of both state and action contexts.

        Args:
            x (Union[np.ndarray, torch.Tensor]): The new state observation(s).
            model_kwargs (Optional[Dict]): Keyword arguments for the model's sample function.
            return_output (bool): Whether to return the model output. Defaults to True.

        Returns:
            Tuple[Union[np.ndarray, torch.Tensor], Optional[ModelOutput], Dict[str, Any]]:
                A tuple of the sampled output, the model forward pass output, and the sample metadata.
        """
        if model_kwargs is None:
            model_kwargs = {}

        if not isinstance(x, list):
            x = [x]

        for _x in x:
            state_context, action_context = self.update_contexts(state=_x)

        model_input = {
            "x": state_context[None, ...],
            "prev_actions": action_context[None, ...],
        }

        output, model_out, metadata = self.model.sample(
            **model_input, temperature=self.temperature, **model_kwargs
        )

        new_action = output.squeeze(0)
        self.update_contexts(action=new_action)
        final_action = (
            output.squeeze((0, 1)).cpu().numpy()
            if self.use_tensors
            else output.squeeze((0, 1))
        )

        return (
            final_action,
            model_out if return_output else None,
            metadata,
        )
