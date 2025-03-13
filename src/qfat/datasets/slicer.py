import logging

import numpy as np
from torch.utils.data import Dataset

from qfat.datasets.dataset import Trajectory, TrajectoryDataset

logger = logging.getLogger(__name__)


class SlicedTrajectoryDataset(Dataset):
    """Adapted from https://github.com/jayLEE0301/vq_bet_official"""

    def __init__(
        self,
        dataset: TrajectoryDataset,
        window: int,
        action_horizon: int = 0,
        min_future_sep: int = 0,
        future_seq_len: int = 0,
        only_sample_tail: bool = False,
    ) -> None:
        """
        Initializes the SlicedTrajectoryDataset.

        This class slices a trajectory dataset into unique (but possibly overlapping)
        sequences of length `window`. If `window` is a list, it will randomly choose
        a window size from that list for each trajectory.

        Args:
            dataset (TrajectoryDataset):
                A trajectory dataset.
                Must implement:
                    - dataset.get_trajectory_len(i)
                    - dataset[i] -> returns a Trajectory object.
            window (int): An integer or a list of integers representing the number of timesteps
                to include in each slice.
            action_horizon (int):
                The number of additional actions concatenated at each timestep.
                E.g., if set to 2, then [action_t, action_t+1, action_t+2] get
                flattened into the action vector at index `t`.
            min_future_sep (int):
                Minimum separation between the current window and the start of
                the future goal sequence.
            future_seq_len (int):
                Number of timesteps in the future goal sequence.
            only_sample_tail (bool):
                If True, always use the final `future_seq_len` steps of the trajectory
                as the future goal.
        """

        self.dataset = dataset
        self.window = window

        self.action_horizon = action_horizon
        self.min_future_sep = min_future_sep
        self.future_seq_len = future_seq_len
        self.only_sample_tail = only_sample_tail

        self.slices = []
        self._create_slices()

    def _create_slices(self) -> None:
        """
        For each trajectory i of length T:
        1. Slices of size 1 ... window_size (but not exceeding T),
            all starting at index 0.
        2. Then sliding windows of size `window_size` for the rest
            of the trajectory (start in [1..(T - window_size)]).
        """
        for i in range(len(self.dataset)):
            T = self._get_trajectory_len(i)
            for w in range(1, min(T, self.window) + 1):
                if w + self.action_horizon <= T:
                    self.slices.append((i, 0, w))
            if T > self.window:
                max_start = T - self.window - self.action_horizon + 1
                for start in range(1, max_start):
                    end = start + self.window
                    self.slices.append((i, start, end))

    def _get_trajectory_len(self, idx: int) -> int:
        """Helper to get the length of the `idx`th trajectory."""
        return self.dataset.get_trajectory_len(idx)

    def _slice_with_horizon(
        self, arr: np.ndarray, start: int, end: int, horizon: int
    ) -> np.ndarray:
        """
        Slice (and optionally flatten) a time-series array from `start` to `end`,
        each step concatenating an additional `horizon` actions.

        Args:
            arr (np.ndarray): The array to slice of shape (T, D) typically.
            start (int): Start index.
            end (int): End index (non-inclusive).
            horizon (int): Number of future timesteps to include at each step.

        Returns:
            np.ndarray: Sliced and possibly flattened array.
        """
        if horizon > 0:
            arr_list = []
            for t in range(start, end):
                arr_slice = arr[t : t + 1 + horizon, :].flatten()
                arr_list.append(arr_slice)
            return np.stack(arr_list, axis=0)
        else:
            return arr[start:end]

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> Trajectory:
        """
        Retrieve a single sliced window (and possibly horizon) from the dataset.
        """
        i, start, end = self.slices[idx]
        traj = self.dataset[i]

        states = traj.states[start:end]
        actions = self._slice_with_horizon(
            arr=traj.actions, start=start, end=end, horizon=self.action_horizon
        )

        prev_actions = None
        if traj.prev_actions is not None:
            prev_actions = self._slice_with_horizon(
                arr=traj.prev_actions, start=start, end=end, horizon=self.action_horizon
            )

        goals = None
        if self.dataset.include_goals:
            T = self.dataset.get_trajectory_len(i)
            valid_start_range = (end + self.min_future_sep, T - self.future_seq_len)

            if valid_start_range[0] < valid_start_range[1]:
                if self.only_sample_tail:
                    # Always pick the last N steps as goals
                    goals = traj.goals[-self.future_seq_len :]
                else:
                    # Sample a random future start
                    future_start = np.random.randint(*valid_start_range)
                    future_end = future_start + self.future_seq_len
                    goals = traj.goals[future_start:future_end]
            else:
                # If there's not enough future, fallback to the last N steps
                goals = traj.goals[-self.future_seq_len :]

        return Trajectory(
            states=states, actions=actions, prev_actions=prev_actions, goals=goals
        )
