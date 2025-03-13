from typing import List, Union

from qfat.datasets.dataset import Trajectory, TrajectoryDataset
from qfat.environments.multi_route import multi_route


class MultiPathTrajectoryDataset(TrajectoryDataset):
    """Adapted from https://github.com/jayLEE0301/vq_bet_official"""

    def __init__(
        self,
        num_samples: int = 200_000,
        noise_scale: Union[float, List[List[float]]] = 0.25,
        **kwargs,
    ):
        super().__init__()
        path_generator = multi_route.PathGenerator(
            waypoints=multi_route.MULTI_PATH_WAYPOINTS_1,
            step_size=1,
            num_draws=100,
            noise_scale=noise_scale,
        )
        self._trajectories = path_generator.get_sequence_dataset(
            num_paths=num_samples, probabilities=multi_route.PATH_PROBS_1
        )

    @property
    def trajectories(self) -> List[Trajectory]:
        return self._trajectories

    @property
    def include_goals(self) -> bool:
        return False
