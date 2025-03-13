from pathlib import Path
from typing import List, Optional

import numpy as np
import zarr

from qfat.constants import KITCHEN_DATA_PATH
from qfat.datasets.dataset import Trajectory, TrajectoryDataset
from qfat.datasets.transform import TrajectoryTransform
from qfat.normalizer.normalizer import MeanStdNormalizer


class KitchenTrajectoryDataset(TrajectoryDataset):
    """Adapted from https://github.com/jayLEE0301/vq_bet_official"""

    def __init__(
        self,
        data_dir: str = str(KITCHEN_DATA_PATH),
        transforms: Optional[List[TrajectoryTransform]] = None,
        include_goals: bool = False,
        stats_path: str = str(KITCHEN_DATA_PATH / "data_stats.json"),
        mode: Optional[str] = None,
        normalize: bool = True,
        include_prev_actions: bool = False,
    ):
        super().__init__()
        self._include_goals = include_goals
        self._data_dir = Path(data_dir)
        self.mode = mode
        self.transforms = transforms

        self._actions = np.load(
            self._data_dir / "actions_seq.npy", mmap_mode="r"
        ).swapaxes(0, 1)
        self._mask = np.load(
            self._data_dir / "existence_mask.npy", mmap_mode="r"
        ).swapaxes(0, 1)
        self._labels = np.load(
            self._data_dir / "onehot_goals.npy", mmap_mode="r"
        ).swapaxes(0, 1)
        self._stats_path = stats_path

        if self.mode == "image":
            self._states = zarr.open_group(
                KITCHEN_DATA_PATH / "precomputed_images.zarr", mode="r"
            )
            self._states = {key: self._states[key][:] for key in self._states.keys()}
        else:
            self._states = np.load(
                self._data_dir / "observations_seq.npy", mmap_mode="r"
            ).swapaxes(0, 1)[..., :30]
        self.num_trajectories = self._mask.shape[0]
        self.include_prev_actions = include_prev_actions
        self.normalizer = MeanStdNormalizer(normalize_actions_flag=True)
        self._normalize = normalize
        if self._normalize:
            self._compute_or_load_normalization_stats()

    def _compute_or_load_normalization_stats(self):
        """compute the normalization stats and save them."""
        stats_file = Path(self._stats_path)
        self._compute_normalization_stats()
        self.normalizer.save_stats(stats_file)

    def _compute_normalization_stats(self):
        """
        Compute min/max (or other stats) across all states/actions/goals
        (if desired) for the entire dataset.

        Note: If your dataset is very large, you may need a more memory-efficient
        approach than simply concatenating all states.
        """
        if self.mode == "image":
            all_actions = []
            all_goals = [] if self._include_goals else None

            for i in range(self.num_trajectories):
                mask_i = self._mask[i, :]
                valid_len = int(mask_i.sum().item())
                actions = self._actions[i, :valid_len, :]
                all_actions.append(actions)
                if self._include_goals:
                    goals = self._states[i, :valid_len]
                    all_goals.append(goals)

            all_actions = np.concatenate(all_actions, axis=0)
            if self._include_goals and len(all_goals) > 0:
                all_goals = np.concatenate(all_goals, axis=0)
            else:
                all_goals = None

            self.normalizer.update_stats(
                None,  # no state normalization
                all_actions,
                None,  # no state normalization
            )
        else:
            all_states = []
            all_actions = []
            all_goals = [] if self._include_goals else None

            for i in range(self.num_trajectories):
                mask_i = self._mask[i, :]
                valid_len = int(mask_i.sum().item())

                states = self._states[i, :valid_len, :]
                all_states.append(states)

                actions = self._actions[i, :valid_len, :]
                all_actions.append(actions)

                if self._include_goals:
                    goals = self._states[i, :valid_len, :]
                    all_goals.append(goals)

            all_states = np.concatenate(all_states, axis=0)
            all_actions = np.concatenate(all_actions, axis=0)
            if self._include_goals and len(all_goals) > 0:
                all_goals = np.concatenate(all_goals, axis=0)
            else:
                all_goals = None

            self.normalizer.update_stats(
                all_states,
                all_actions,
                goals=all_goals,
            )

    @property
    def trajectories(self) -> None:
        raise NotImplementedError(
            "The dataset is too big to access all trajectories at once."
        )

    @property
    def include_goals(self) -> bool:
        return self._include_goals

    def __len__(self):
        return self.num_trajectories

    def __getitem__(self, index: int) -> Trajectory:
        if index >= self.num_trajectories or index < 0:
            raise IndexError(
                f"Index {index} out of range for {self.num_trajectories} trajectories."
            )

        mask_i = self._mask[index, :]
        valid_len = int(mask_i.sum().item())

        actions = self._actions[index, :valid_len, :]
        labels = self._labels[index, :valid_len]

        # Get states or images depending on mode
        if self.mode == "image":
            episode_images = self._states[f"episode_{index}"][:]
            states = episode_images[:valid_len]
            goals = states if self._include_goals else None
        else:
            states = self._states[index, :valid_len, :]
            goals = states if self._include_goals else None

        if self._normalize:
            if self.mode != "image":
                states = self.normalizer.normalize_state(states)
                if self._include_goals and goals is not None:
                    goals = self.normalizer.normalize_goal(goals)
            actions = self.normalizer.normalize_action(actions)

        prev_actions = None
        if self.include_prev_actions:
            prev_actions = np.zeros_like(actions)
            prev_actions[1:] = actions[:-1]
            prev_actions = prev_actions.astype(np.float32)

        trajectory = Trajectory(
            states=states,
            actions=actions,
            prev_actions=prev_actions,
            labels=labels,
            goals=goals,
        )

        if self.transforms is not None:
            for transform in self.transforms:
                trajectory = transform(trajectory=trajectory)

        return trajectory


if __name__ == "__main__":
    ds = KitchenTrajectoryDataset(include_goals=True)
