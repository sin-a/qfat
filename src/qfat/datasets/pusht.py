from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr
from numpy.typing import NDArray

from qfat.constants import PUSHT_DATA_PATH
from qfat.datasets.dataset import Trajectory, TrajectoryDataset
from qfat.datasets.transform import (
    TrajectoryTransform,
)
from qfat.normalizer.normalizer import MinMaxNormalizer


class PushTTrajectoryDataset(TrajectoryDataset):
    """Adapted from https://github.com/jayLEE0301/vq_bet_official"""

    def __init__(
        self,
        data_dir: str = str(PUSHT_DATA_PATH),
        transforms: Optional[List[TrajectoryTransform]] = None,
        stats_path: str = str(PUSHT_DATA_PATH / "data_stats.json"),
        mode: str = "keypoints",
        embedding_path: str = str(PUSHT_DATA_PATH / "resnet_embeddings"),
        include_goals: bool = False,
        include_prev_actions: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir (str): Path to the dataset directory.
            transforms (Optional[List[TrajectoryTransform]]): List of transforms to apply to the trajectories.
            stats_path (str): Path to save or load normalization stats.
            mode (str): Input mode for the dataset - "keypoints", "image", or "embeddings".
            embedding_path (str): Path to the directory containing precomputed embeddings.
            include_goals (bool): Whether to include goals in the trajectories.
        """
        super().__init__(transforms)

        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.stats_path = Path(stats_path) if stats_path is not None else None
        self.mode = mode
        self.embedding_path = Path(embedding_path)
        self._include_goals = include_goals
        self.include_prev_actions = include_prev_actions
        if self.mode not in {"keypoints", "image", "embeddings"}:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Choose 'keypoints', 'image', or 'embeddings'."
            )

        src_root = self._load_zarr_data()
        meta, data = self._load_metadata_and_data(src_root)

        self.normalizer = MinMaxNormalizer(normalize_actions_flag=True)

        if self.mode == "keypoints":
            all_states, all_actions, all_goals = self._collect_data_for_normalization(
                data, meta
            )
            self.normalizer.update_stats(all_states, all_actions, goals=all_goals)
        else:
            if self.mode == "embeddings":
                self._check_embeddings_exist()
            all_actions = self._collect_actions_for_normalization(data, meta)
            self.normalizer.update_stats(states=None, actions=all_actions, goals=None)

        if self.stats_path:
            self.normalizer.save_stats(self.stats_path)

        self._meta = meta
        self._data = data
        self._trajectories = self._create_trajectories()

    @property
    def trajectories(self):
        return self._trajectories

    @property
    def include_goals(self):
        return self._include_goals

    def _load_zarr_data(self):
        """Load the root Zarr group from the data directory."""
        return zarr.group(self.data_dir / "pusht_cchi_v7_replay.zarr")

    def _load_metadata_and_data(
        self, src_root
    ) -> Tuple[Dict[str, NDArray], Dict[str, NDArray]]:
        """Load metadata and data from the Zarr dataset."""
        meta = {
            key: np.array(value) if len(value.shape) == 0 else value[:]
            for key, value in src_root["meta"].items()
        }
        data = {key: src_root["data"][key] for key in src_root["data"].keys()}
        return meta, data

    def _check_embeddings_exist(self):
        """Check if the embeddings directory exists and contains trajectory files."""
        if not self.embedding_path.exists() or not any(
            self.embedding_path.glob("*.npy")
        ):
            raise FileNotFoundError(
                f"Embeddings not found in '{self.embedding_path}'. Please run the embedding generation script first."
            )

    def _collect_data_for_normalization(
        self, data: Dict[str, Any], meta: Dict[str, NDArray]
    ) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
        """Collect states, actions, and goals for normalization."""
        all_states, all_actions, all_goals = [], [], []
        start = 0
        if self.mode != "keypoints":
            raise ValueError(
                "This method should only be called when mode is keypoints."
            )

        obs = self._extract_keypoints(data)
        goals = obs
        for end in meta["episode_ends"]:
            if (300 - (end - start)) <= 0:
                continue
            all_states.append(obs[start:end])
            all_actions.append(data["action"][start:end])
            if self.include_goals:
                all_goals.append(goals[start:end])
            start = end

        return (
            np.concatenate(all_states, axis=0),
            np.concatenate(all_actions, axis=0),
            np.concatenate(all_goals, axis=0) if self.include_goals else None,
        )

    def _collect_actions_for_normalization(
        self, data: Dict[str, Any], meta: Dict[str, NDArray]
    ) -> NDArray:
        """Collect actions for normalization."""
        all_actions = []
        start = 0

        for end in meta["episode_ends"]:
            if (300 - (end - start)) <= 0:
                continue
            all_actions.append(data["action"][start:end])
            start = end

        return np.concatenate(all_actions, axis=0)

    def _create_trajectories(self) -> List[Trajectory]:
        """Generate a list of Trajectory instances."""
        trajectories = []
        start = 0
        raw_goals = None
        for idx, end in enumerate(self._meta["episode_ends"]):
            if (300 - (end - start)) <= 0:
                continue

            if self.mode == "keypoints":
                obs = self._extract_keypoints(self._data)[start:end]
                obs = self.normalizer.normalize_state(obs)
                goals = obs if self.include_goals else None
                if goals is not None:
                    goals = self.normalizer.normalize_goal(goals)
            elif self.mode == "image":
                obs = self._data["img"][start:end]
                goals = obs if self.include_goals else None
            elif self.mode == "embeddings":
                embedding_file = self.embedding_path / f"trajectory_{idx}.npy"
                if not embedding_file.exists():
                    raise FileNotFoundError(
                        f"Embedding file '{embedding_file}' not found. Please generate embeddings first."
                    )
                obs = np.load(embedding_file)
                goals = obs if self.include_goals else None
            raw_goals = (
                self._extract_block_state(self._data)[start:end]
                if self.include_goals
                else None
            )
            actions = self.normalizer.normalize_action(self._data["action"][start:end])
            prev_actions = None
            if self.include_prev_actions:
                prev_actions = np.zeros_like(actions)
                prev_actions[1:] = actions[:-1]
                prev_actions = prev_actions.astype(np.float32)
            traj = Trajectory(
                states=obs.astype(np.float32),
                actions=actions.astype(np.float32),
                goals=goals,
                raw_goals=raw_goals,
                prev_actions=prev_actions,
            )
            trajectories.append(traj)
            start = end

        return trajectories

    def _extract_keypoints(self, data: Dict[str, NDArray]) -> NDArray:
        """Extract keypoint observations, concatenated with agent position."""
        agent_pos = np.array(data["state"][:, :2])
        return np.concatenate(
            [
                np.array(data["keypoint"]).reshape(data["keypoint"].shape[0], -1),
                agent_pos,
            ],
            axis=-1,
        )

    def _extract_block_state(self, data: Dict[str, NDArray]):
        block_state = np.array(data["state"][:, 2:])  # pos x, pos y, orientation
        return block_state


if __name__ == "__main__":
    import matplotlib.pyplot as plt  # Noqa

    ds = PushTTrajectoryDataset(mode="image")
