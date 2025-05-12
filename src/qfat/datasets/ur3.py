from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from qfat.constants import UR3_DATA_PATH
from qfat.datasets.dataset import Trajectory, TrajectoryDataset
from qfat.datasets.transform import TrajectoryTransform
from qfat.normalizer.normalizer import MinMaxNormalizer


class UR3TrajectoryDataset(TrajectoryDataset):
    """Adapted from https://github.com/jayLEE0301/vq_bet_official"""

    def __init__(
        self,
        data_dir: str = str(UR3_DATA_PATH),
        transforms: Optional[List[TrajectoryTransform]] = None,
        stats_path: str = str(UR3_DATA_PATH / "data_stats.json"),
        include_goals: bool = False,
    ):
        """
        Initialize the AntTrajectoryDataset.

        Args:
            data_dir (str): Directory where the ant trajectory data is located.
            transforms (Optional[List[TrajectoryTransform]]): List of transforms to apply to trajectories.
            stats_path (Optional[str]): Path to save or load normalizer stats.
            include_goals (bool): Whether to include goals in the trajectories.
        """
        super().__init__(transforms)

        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.stats_path = Path(stats_path) if stats_path else None
        self._include_goals = include_goals

        actions, states, mask = self._load_data()
        self.normalizer = MinMaxNormalizer()

        all_states, all_actions, all_goals = self._collect_data_for_normalization(
            states, actions, mask
        )
        self.normalizer.update_stats(all_states, all_actions, all_goals)

        if self.stats_path:
            self.normalizer.save_stats(self.stats_path)

        self._trajectories = self._create_trajectories(states, actions, mask)

    def _load_data(self) -> Tuple[NDArray, NDArray, NDArray]:
        """Load data from files."""
        actions = np.load(self.data_dir / "data_act.npy")
        states = np.load(self.data_dir / "data_obs.npy")
        mask = np.load(self.data_dir / "data_msk.npy")
        return actions, states, mask

    def _collect_data_for_normalization(
        self, states: NDArray, actions: NDArray, mask: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """Collect valid states and actions for normalization."""
        all_states, all_actions, all_goals = [], [], []
        for i in range(states.shape[0]):
            valid_len = int(mask[i].sum().item())
            all_states.append(states[i, :valid_len, :])
            all_actions.append(actions[i, :valid_len, :])
            goal = states[i, :valid_len].copy()
            goal[:, :2] = 0
            all_goals.append(goal)

        return (
            np.concatenate(all_states, axis=0),
            np.concatenate(all_actions, axis=0),
            np.concatenate(all_goals, axis=0) if self.include_goals else None,
        )

    def _create_trajectories(
        self, states: NDArray, actions: NDArray, mask: NDArray
    ) -> List[Trajectory]:
        """Generate normalized Trajectories from the dataset."""
        trajectories = []
        for i in range(states.shape[0]):
            valid_len = int(mask[i].sum().item())
            norm_states = self.normalizer.normalize_state(states[i, :valid_len, :])
            norm_actions = self.normalizer.normalize_action(actions[i, :valid_len, :])
            goals = None
            if self.include_goals:
                goals = states[i, :valid_len, :].copy()
                goals[:, :2] = 0
                goals = self.normalizer.normalize_goal(goals)
            trajectories.append(
                Trajectory(
                    states=norm_states.astype(np.float32),
                    actions=norm_actions.astype(np.float32),
                    goals=goals.astype(np.float32) if goals is not None else None,
                )
            )
        return trajectories

    @property
    def trajectories(self) -> List[Trajectory]:
        return self._trajectories

    @property
    def include_goals(self):
        return self._include_goals

    def __getitem__(self, idx: int) -> Trajectory:
        """Retrieve a normalized trajectory, optionally applying transforms."""
        trajectory = self._trajectories[idx]
        if self.transforms:
            for transform in self.transforms:
                trajectory = transform(trajectory=trajectory)
        return trajectory


if __name__ == "__main__":
    ds = UR3TrajectoryDataset(include_goals=True)
