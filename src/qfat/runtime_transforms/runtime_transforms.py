import math
from abc import ABC, abstractmethod
from typing import Optional

import einops
import torch
from torchvision import transforms

import wandb
from qfat.datasets.dataset import Batch


class RuntimeTransform(ABC):
    """Applied to every training batch before being passed to the model"""

    @abstractmethod
    def __call__(self, batch: Batch, epoch: int) -> Batch:
        """Transforms a training batch, and the transformation can depend on the epoch"""
        pass


class CosineGaussianPerturbation(RuntimeTransform):
    """Adds Gaussian noise to states and actions in a batch with optional cosine decay."""

    def __init__(
        self,
        state_initial_noise: Optional[float] = None,
        state_final_noise: Optional[float] = None,
        state_input_index: Optional[int] = None,
        action_initial_noise: Optional[float] = None,
        action_final_noise: Optional[float] = None,
        action_input_index: Optional[int] = None,
        log_noise: bool = True,
        max_epoch: Optional[int] = None,
        fixed_noise: bool = False,
    ) -> None:
        """Instantiates a Gaussian perturbation transform with optional cosine decay.

        Args:
            state_initial_noise (float): Initial Gaussian scale for state perturbations.
            state_final_noise (float): Final Gaussian scale for state perturbations.
            state_input_index (int): Up to which index in the state input vector noise is applied.
            action_initial_noise (float): Initial Gaussian scale for action perturbations.
            action_final_noise (float): Final Gaussian scale for action perturbations.
            action_input_index (int): Up to which index in the action input vector noise is applied.
            log_noise (bool): Whether to log the noise scale during training.
            max_epoch (int): The maximum number of epochs for cosine decay.
            fixed_noise (bool): Whether to keep the noise level fixed (disable decay).
        """
        self.state_params = {
            "initial_noise": state_initial_noise,
            "final_noise": state_final_noise,
            "input_index": state_input_index,
        }
        self.action_params = {
            "initial_noise": action_initial_noise,
            "final_noise": action_final_noise,
            "input_index": action_input_index,
        }
        self.log_noise = log_noise
        self.max_epoch = max_epoch
        self.fixed_noise = fixed_noise
        self.epoch_counter = {"state": -1, "action": -1}

    def _compute_noise_scale(self, epoch: int, initial: float, final: float) -> float:
        """Computes the noise scale using cosine schedule or fixed scale."""
        if self.fixed_noise or self.max_epoch is None:
            return initial
        cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / self.max_epoch))
        return final + (initial - final) * cosine_decay

    def _transform(
        self, x: torch.Tensor, epoch: int, params: dict, noise_type: str
    ) -> torch.Tensor:
        """Applies Gaussian noise transformation to a tensor."""
        if (
            params["initial_noise"] is None
            or self.max_epoch is None
            and not self.fixed_noise
        ):
            return x

        _x = x
        if params["input_index"] is not None:
            _x = x[..., : params["input_index"]]

        noise_scale = self._compute_noise_scale(
            epoch,
            params["initial_noise"],
            params["final_noise"] or params["initial_noise"],
        )
        _x += noise_scale * torch.normal(mean=0, std=torch.ones_like(_x)).to(x.device)

        if epoch > self.epoch_counter[noise_type] and self.log_noise:
            wandb.log({f"train/{noise_type}_noise_scale": noise_scale})
            self.epoch_counter[noise_type] += 1

        return x

    def __call__(self, batch: Batch, epoch: int) -> Batch:
        """Applies the transformation to states and actions in the batch."""
        transform_fn = {}

        if self.state_params["initial_noise"] is not None:
            transform_fn["x"] = lambda x: self._transform(
                x, epoch=epoch, params=self.state_params, noise_type="state"
            )

        if self.action_params["initial_noise"] is not None:
            transform_fn["y"] = lambda x: self._transform(
                x, epoch=epoch, params=self.action_params, noise_type="action"
            )

        return batch.transform(transform_fns=transform_fn)


class AnnealedGaussianPerturbation(RuntimeTransform):
    """Adds exponentially decaying Gaussian noise to states and actions in a batch."""

    def __init__(
        self,
        state_noise_scale: Optional[float] = None,
        state_input_index: Optional[int] = None,
        action_noise_scale: Optional[float] = None,
        action_input_index: Optional[int] = None,
        log_noise: bool = True,
    ) -> None:
        """Instantiates an exponentially decaying Gaussian perturbation transform.

        Args:
            state_noise_scale (float): The initial Gaussian scale for state perturbations.
            state_decay_factor (int): The exponential decay factor for states.
            state_input_index (int): Up to which index in the state input vector noise is applied.
            action_noise_scale (float): The initial Gaussian scale for action perturbations.
            action_decay_factor (int): The exponential decay factor for actions.
            action_input_index (int): Up to which index in the action input vector noise is applied.
            log_noise (bool): Whether to log the noise scale during training.
        """
        self.state_params = {
            "noise_scale": state_noise_scale,
            "input_index": state_input_index,
        }
        self.action_params = {
            "noise_scale": action_noise_scale,
            "input_index": action_input_index,
        }
        self.log_noise = log_noise
        self.epoch_counter = {"state": -1, "action": -1}

    def _transform(
        self, x: torch.Tensor, epoch: int, params: dict, noise_type: str
    ) -> torch.Tensor:
        """Applies Gaussian noise transformation to a tensor."""
        if params["noise_scale"] is None or params["decay_factor"] is None:
            return x

        _x = x
        if params["input_index"] is not None:
            _x = x[..., : params["input_index"]]

        noise_scale = params["noise_scale"] * (params["decay_factor"] ** epoch)
        _x += noise_scale * torch.normal(
            mean=0, std=params["noise_scale"] * torch.ones_like(_x)
        ).to(x.device)

        if epoch > self.epoch_counter[noise_type] and self.log_noise:
            wandb.log({f"train/{noise_type}_noise_scale": noise_scale})
            self.epoch_counter[noise_type] += 1

        return x

    def __call__(self, batch: Batch, epoch: int) -> Batch:
        """Applies the transformation to states and actions in the batch."""
        transform_fn = {}

        if self.state_params["noise_scale"] is not None:
            transform_fn["x"] = lambda x: self._transform(
                x, epoch=epoch, params=self.state_params, noise_type="state"
            )

        if self.action_params["noise_scale"] is not None:
            transform_fn["y"] = lambda x: self._transform(
                x, epoch=epoch, params=self.action_params, noise_type="action"
            )

        return batch.transform(transform_fns=transform_fn)


class ImageAugmentationTransform(RuntimeTransform):
    def __init__(self, image_size: int, prob: float = 0.5):
        """
        Args:
            image_size (int): The desired size for random cropping.
            prob (float): Probability of applying each augmentation.
        """
        self.image_size = image_size
        self.prob = prob

        self.augmentations = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2
                        )
                    ],
                    p=self.prob,
                ),
                transforms.RandomApply(
                    [transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0))],
                    p=self.prob,
                ),
                transforms.RandomApply(
                    [transforms.RandomGrayscale(p=1.0)],
                    p=self.prob,
                ),
            ]
        )

        self.pipeline = transforms.Compose(
            [
                transforms.Lambda(
                    lambda x: torch.stack([self.augmentations(x_) for x_ in x])
                )
            ]
        )

    def _augment_tensor(self, x):
        """Rearrange, augment via self.pipeline, and reshape back."""
        if x is None:
            return None

        device = x.device
        b, t, h, w, c = x.shape

        x = einops.rearrange(x, "b t h w c -> (b t) c h w")
        x = self.pipeline(x)
        x = x.view(b, t, *x.shape[1:]).permute(0, 1, 3, 4, 2)
        return x.to(device)

    def __call__(self, batch: Batch, epoch: int) -> Batch:
        """Applies augmentations to `batch.x` and `batch.conditional_seq` if available."""
        batch.x = self._augment_tensor(batch.x)
        if batch.conditional_seq is not None:
            batch.conditional_seq = self._augment_tensor(batch.conditional_seq)
        return batch
