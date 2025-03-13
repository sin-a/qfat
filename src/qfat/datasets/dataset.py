from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.utils
import torch.utils.data
from numpy.typing import NDArray
from torch.utils.data import Dataset, Subset


@dataclass
class Trajectory:
    states: NDArray  # (sequence_length, state_dim)
    actions: NDArray  # (sequence_length, action_dim)
    prev_actions: Optional[NDArray] = None  # (sequence_length, action_dim)
    goals: Optional[NDArray] = None  # (sequence_length, *goal_dim)
    raw_goals: Optional[NDArray] = None  # (sequence_length, raw_goal_dim)
    labels: Optional[NDArray] = None  # (sequenc_length, labels_onehot_dim)

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx) -> "Trajectory":
        """Allows indexing across all tensor attributes simultaneously, handling None attributes gracefully."""
        field_values = {}
        for f in fields(self):
            attr_value = getattr(self, f.name)
            if attr_value is not None:
                field_values[f.name] = attr_value[idx]
            else:
                field_values[f.name] = None
        return Trajectory(**field_values)

    def transform(
        self, transform_fns: Dict[str, Callable[[NDArray], NDArray]]
    ) -> "Trajectory":
        """Applies transformation functions to fields specified in the transform_fns dict."""
        valid_fields = {f.name for f in fields(self)}
        if not set(transform_fns.keys()).issubset(valid_fields):
            raise ValueError(
                f"Invalid field keys in transform_fns. Valid keys are: {valid_fields}"
            )

        transformed_data = {}
        for field in valid_fields:
            if field in transform_fns:
                transformed_data[field] = transform_fns[field](getattr(self, field))
            else:
                transformed_data[field] = getattr(self, field)

        return Trajectory(**transformed_data)


class TrajectoryDataset(Dataset, metaclass=ABCMeta):
    """Adapted from https://github.com/jayLEE0301/vq_bet_official"""

    def __init__(
        self,
        transforms=None,
    ) -> None:
        super().__init__()
        self.transforms = transforms

    @property
    @abstractmethod
    def trajectories(self) -> List[Trajectory]:
        """A List[Trajectory] attribute"""
        pass

    @property
    @abstractmethod
    def include_goals(self) -> bool:
        """Wether the trajectory contains the goals defined or not"""
        pass

    def __len__(self) -> int:
        """Return the total number of trajectories."""
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Trajectory:
        """Retrieves the trajectory at a specific index, optionally applying transforms to it"""
        trajectory = self.trajectories[idx]
        if self.transforms is not None:
            for transform in self.transforms:
                trajectory = transform(trajectory=trajectory)

        return trajectory

    def get_trajectory_len(self, idx) -> int:
        """Retrieve the trajectory length at the given index.

        Args:
            idx (int): Index of the trajectory to retrieve.

        Returns:
            The trajectory length
        """
        return len(self.trajectories[idx])

    def split(
        self, **kwargs
    ) -> Tuple["SubsetTrajectoryDataset", "SubsetTrajectoryDataset"]:
        """Splits the data randomly, uses the same **kwargs as torch.utils.data.random_split."""
        ds1, ds2 = torch.utils.data.random_split(self, **kwargs)
        return SubsetTrajectoryDataset(
            ds1, transforms=self.transforms
        ), SubsetTrajectoryDataset(ds2, transforms=self.transforms)


class SubsetTrajectoryDataset(TrajectoryDataset):
    def __init__(self, subset: Subset, transforms=None):
        super().__init__(transforms)
        self.subset = subset

    def __len__(self):
        return len(self.subset.indices)

    def __getitem__(self, idx: int) -> Trajectory:
        """Retrieves the trajectory at a specific index"""
        trajectory = self.subset.dataset[self.subset.indices[idx]]
        return trajectory

    @property
    def trajectories(self) -> None:
        raise ValueError("Can't access the raw tajectories list in a subset dataset.")

    @property
    def include_goals(self):
        return self.subset.dataset.include_goals

    def get_trajectory_len(self, idx) -> int:
        return len(self.subset.dataset[self.subset.indices[idx]])


@dataclass
class Batch:
    x: torch.Tensor
    y: Optional[torch.Tensor] = None
    validity_mask: Optional[torch.Tensor] = None
    conditional_seq: Optional[torch.Tensor] = None
    prev_actions: Optional[Any] = None

    def to(self, device: str) -> None:
        """Moves all tensor attributes to the specified device in-place."""
        for f in fields(self):
            attr = getattr(self, f.name)
            if attr is not None and isinstance(attr, torch.Tensor):
                setattr(self, f.name, attr.to(device))

    def transform(
        self, transform_fns: Dict[str, Callable[[NDArray], NDArray]]
    ) -> "Batch":
        """Applies transformation functions to fields specified in the transform_fns dict."""
        valid_fields = {f.name for f in fields(self)}
        if not set(transform_fns.keys()).issubset(valid_fields):
            raise ValueError(
                f"Invalid field keys in transform_fns. Valid keys are: {valid_fields}"
            )

        transformed_data = {}
        for field in valid_fields:
            if field in transform_fns:
                transformed_data[field] = transform_fns[field](getattr(self, field))
            else:
                transformed_data[field] = getattr(self, field)

        return Batch(**transformed_data)


def collate_policy(batch: List[Trajectory]) -> Batch:
    """A collate function that batches the trajectories and pads them to the longest trajectory length.

    Args:
        batch (List[Trajectory]): A list of trajectories to be batched.
            This comes from the DataLoader sampling indexes and putting the trajectories
            in a list.

    Returns:
        Batch: A dataclass (or similar) with:
            - x: padded states
            - y: padded actions
            - prev_actions: padded prev_actions (if any)
            - conditional_seq: padded goals (if any)
            - validity_mask: mask denoting where the sequence is valid (1) and where it is padded (0).
    """
    states_list = [torch.tensor(traj.states, dtype=torch.float32) for traj in batch]
    actions_list = [torch.tensor(traj.actions, dtype=torch.float32) for traj in batch]
    prev_actions_list = [
        torch.tensor(traj.prev_actions, dtype=torch.float32)
        for traj in batch
        if traj.prev_actions is not None
    ]
    goals_list = [
        torch.tensor(traj.goals, dtype=torch.float32)
        for traj in batch
        if traj.goals is not None
    ]

    if len(states_list) == 0 or len(actions_list) == 0:
        raise ValueError("All trajectories are empty. Cannot pad empty sequences.")

    lengths = torch.tensor(
        [states.shape[0] for states in states_list], dtype=torch.int64
    )

    states_padded = torch.nn.utils.rnn.pad_sequence(
        [s.flip(0) for s in states_list], batch_first=True, padding_value=0.0
    ).flip(1)

    actions_padded = torch.nn.utils.rnn.pad_sequence(
        [a.flip(0) for a in actions_list], batch_first=True, padding_value=0.0
    ).flip(1)

    prev_actions_padded = None
    if len(prev_actions_list) > 0:
        prev_actions_padded = torch.nn.utils.rnn.pad_sequence(
            [a.flip(0) for a in prev_actions_list], batch_first=True, padding_value=0.0
        ).flip(1)

    goals_padded = None
    if len(goals_list) > 0:
        goals_padded = torch.nn.utils.rnn.pad_sequence(
            [a.flip(0) for a in goals_list], batch_first=True, padding_value=0.0
        ).flip(1)

    max_length = states_padded.size(1)
    validity_mask = torch.arange(max_length).expand(len(lengths), max_length).flip(
        1
    ) < lengths.unsqueeze(1)

    return Batch(
        x=states_padded,
        y=actions_padded,
        prev_actions=prev_actions_padded,
        validity_mask=validity_mask,
        conditional_seq=goals_padded,
    )


IMPLEMENTED_COLLATE_FNS = {
    "collate_policy": collate_policy,
}
