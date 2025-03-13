import random
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image, ImageEnhance

from qfat.datasets.dataset import Trajectory


class TrajectoryTransform(ABC):
    @abstractmethod
    def __call__(self, trajectory: Trajectory) -> Trajectory:
        pass


class ConcatPrevActionsTransform(TrajectoryTransform):
    def __call__(self, trajectory: Trajectory) -> Trajectory:
        """
        Adds previous actions to the states of a trajectory.

        Args:
            trajectory (Trajectory): The trajectory to transform.

        Returns:
            Trajectory: A new trajectory with states concatenated with previous actions.
        """
        prev_actions = np.zeros_like(trajectory.actions)
        prev_actions[1:] = trajectory.actions[:-1]
        states_with_prev_actions = np.concatenate(
            [trajectory.states, prev_actions], axis=-1
        )
        return Trajectory(
            states=states_with_prev_actions,
            actions=trajectory.actions,
            goals=trajectory.goals,
        )


class ImgAugTransform(TrajectoryTransform):
    def __call__(self, trajectory: Trajectory) -> Trajectory:
        """
        Augments the images that are input to a CNN model.

        Args:
            trajectory (Trajectory): The trajectory to transform.

        Returns:
            Trajectory: A new trajectory with the augmented images.
        """
        augmented_images = []
        for (
            img
        ) in trajectory.states:  # Assuming states are images (episode_len, W, H, 3)
            augmented_img = self._augment_image(img)
            augmented_images.append(augmented_img)

        augmented_images = np.stack(augmented_images)

        return Trajectory(
            states=augmented_images,
            actions=trajectory.actions,
            goals=trajectory.goals,
        )

    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply random color jitter and random cropping + resizing to an image.

        Args:
            img (np.ndarray): The image to augment, shape (W, H, 3).

        Returns:
            np.ndarray: The augmented image, shape (W, H, 3).
        """
        pil_img = Image.fromarray(img.astype("uint8"))  # Convert to PIL image
        # Apply color jitter
        if random.random() < 0.5:
            pil_img = self._apply_color_jitter(pil_img)

        # Apply random cropping and resizing
        if random.random() < 0.5:
            pil_img = self._apply_random_crop_and_resize(
                pil_img, target_size=pil_img.size
            )

        return np.array(pil_img)  # Convert back to NumPy array

    def _apply_color_jitter(self, pil_img: Image.Image) -> Image.Image:
        """
        Apply random color jitter to the PIL image.

        Args:
            pil_img (Image.Image): The input PIL image.

        Returns:
            Image.Image: The color-jittered image.
        """
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        saturation_factor = random.uniform(0.8, 1.2)

        # Apply brightness
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(brightness_factor)

        # Apply contrast
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(contrast_factor)

        # Apply saturation
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(saturation_factor)

        return pil_img

    def _apply_random_crop_and_resize(
        self, pil_img: Image.Image, target_size=(224, 224)
    ) -> Image.Image:
        """
        Apply random cropping and resizing to the PIL image.

        Args:
            pil_img (Image.Image): The input PIL image.
            target_size (tuple): The target size (width, height) for resizing.

        Returns:
            Image.Image: The cropped and resized image.
        """
        width, height = pil_img.size
        crop_width = random.randint(int(width * 0.8), width)
        crop_height = random.randint(int(height * 0.8), height)

        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        right = left + crop_width
        bottom = top + crop_height

        # Crop and resize
        pil_img = pil_img.crop((left, top, right, bottom))
        pil_img = pil_img.resize(target_size, Image.Resampling.NEAREST)

        return pil_img
