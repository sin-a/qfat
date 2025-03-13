from typing import List, Union

import einops
import matplotlib.pyplot as plt
import numpy as np
import omegaconf

from qfat.datasets.dataset import Trajectory

BOUNDS = 10
MULTI_PATH_WAYPOINTS_1 = (
    (
        (0, 0),
        (BOUNDS, 0),
        (BOUNDS, BOUNDS),
        (BOUNDS, 2 * BOUNDS),
        (2 * BOUNDS, 2 * BOUNDS),
    ),
    (
        (0, 0),
        (BOUNDS, BOUNDS),
        (2 * BOUNDS, 2 * BOUNDS),
    ),
    (
        (0, 0),
        (0, BOUNDS),
        (BOUNDS, BOUNDS),
        (2 * BOUNDS, BOUNDS),
        (2 * BOUNDS, 2 * BOUNDS),
    ),
)

PATH_PROBS_1 = (0.35, 0.3, 0.35)
PATH_PROBS_2 = (0.5, 0.0, 0.5)


def get_cmap(n, name="rainbow"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def interpolate(point_1, point_2, num_intermediate, endpoint=True):
    t = np.linspace(0, 1, num_intermediate, endpoint=endpoint).reshape(-1, 1)
    interpolated_array = ((1 - t) * point_1.T) + (t * point_2.T)
    return interpolated_array


class PathGenerator:
    def __init__(
        self,
        waypoints: List[List[tuple]],
        step_size: float,
        num_draws: int = 10,
        noise_scale: Union[float, List[List[float]]] = 0.25,
    ):
        """
        waypoints: List of paths, each path is a list of (x, y) tuples.
        step_size: Step size for interpolation.
        num_draws: Number of random draws when visualizing.
        noise_scale: Either a single float (applied to all paths) or a list of
                     noise vectors, each corresponding to a path. Each noise
                     vector should be a list or array with the same dimension
                     as the path points (e.g., [noise_x, noise_y]).
        """
        self.waypoints = waypoints
        self.step_size = step_size
        self._num_draws = num_draws
        if isinstance(noise_scale, list) or isinstance(
            noise_scale, omegaconf.listconfig.ListConfig
        ):
            assert len(noise_scale) == len(self.waypoints), (
                "Length of noise_scale list must match number of paths (waypoints)."
            )
            self._noise_scale = noise_scale
        else:
            # Apply the same noise_scale to all paths
            self._noise_scale = [noise_scale] * len(self.waypoints)

        self.build_paths()

    def build_paths(self):
        self._paths = []
        for wp in self.waypoints:
            waypoints = np.array(wp)
            final_path = []
            for i in range(len(waypoints) - 1):
                point_1, point_2 = waypoints[i], waypoints[i + 1]
                path_length = np.linalg.norm(point_1 - point_2)
                path_num_steps = max(int(path_length / self.step_size), 1)
                final_path.append(
                    interpolate(
                        point_1,
                        point_2,
                        path_num_steps,
                        endpoint=(i == (len(waypoints) - 2)),
                    )
                )
            final_path = np.concatenate(final_path)
            self._paths.append(final_path)

    def draw(self):
        # Draw the paths a little randomly each time.
        cmap = get_cmap(len(self._paths) + 1)
        for idx, path in enumerate(self._paths):
            path_color = cmap(idx)
            for _ in range(self._num_draws):
                noise_vector = self._noise_scale[idx]
                random_noise = np.random.normal(
                    loc=0.0,
                    scale=[var**0.5 for var in noise_vector],
                    size=path.shape,
                )
                random_path = path + random_noise

                plt.plot(
                    random_path[:, 0], random_path[:, 1], "o-", c=path_color, alpha=0.25
                )
        plt.show()

    def get_random_paths(
        self, num_paths: int, probabilities: Union[List[float], None] = None
    ):
        if probabilities is None:
            probabilities = np.ones(len(self._paths))
        else:
            probabilities = np.array(probabilities)

        assert len(probabilities) == len(self._paths), (
            "Length of probabilities must match number of paths."
        )
        # Normalize the probabilities
        probabilities = probabilities / np.sum(probabilities)
        num_paths_on_each = np.floor(num_paths * probabilities).astype(int)

        paths_dataset = []
        for idx, (num_generated_path, path) in enumerate(
            zip(num_paths_on_each, self._paths)
        ):
            if num_generated_path == 0:
                continue
            noise_vector = self._noise_scale[idx]
            (path_len, dim) = path.shape
            random_noise = np.random.normal(
                loc=0.0,
                scale=[var**0.5 for var in noise_vector],
                size=(num_generated_path, path_len, dim),
            )
            random_paths = (
                einops.repeat(
                    path, "len dim -> batch len dim", batch=num_generated_path
                )
                + random_noise
            )
            paths_dataset.append(random_paths)

        return paths_dataset

    @staticmethod
    def get_obs_action_from_path(paths: np.ndarray):
        """Given path of dimensions batch x len x dimension, convert it to two arrays of obs and actions."""
        obses = paths[:, :-1, :]
        # Find the velocity vector.
        actions = paths[:, 1:, :] - paths[:, :-1, :]
        return obses, actions

    def get_memoryless_dataset(
        self, num_paths: int, probabilities: Union[List[float], None] = None
    ):
        all_paths = self.get_random_paths(num_paths, probabilities)
        all_obses, all_actions = [], []
        for path in all_paths:
            obses, actions = PathGenerator.get_obs_action_from_path(path)
            all_obses.append(
                einops.rearrange(obses, "batch len dim -> (batch len) dim")
            )
            all_actions.append(
                einops.rearrange(actions, "batch len dim -> (batch len) dim")
            )

        # This should have dimensions dataset_size (sum path lengths) x dim
        full_obs_dataset = np.concatenate(all_obses)
        # Calculate the next move.
        full_action_dataset = np.concatenate(all_actions)
        return (full_obs_dataset, full_action_dataset)

    def get_sequence_dataset(
        self, num_paths: int, probabilities: Union[List[float], None] = None
    ) -> List[Trajectory]:
        all_paths = self.get_random_paths(num_paths, probabilities)
        trajectories = []
        for path in all_paths:
            obses, actions = PathGenerator.get_obs_action_from_path(path)
            trajectories.extend(
                [
                    Trajectory(states=obses[i, :, :], actions=actions[i, :, :])
                    for i in range(len(obses))
                ]
            )
        return trajectories


if __name__ == "__main__":
    # Example usage with different noise vectors for each path
    noise_scales = [
        [0.65, 0.1],  # Noise for first path (x and y)
        [0.1, 0.1],  # Noise for third path
        [0.1, 0.65],  # Noise for second path
    ]

    gen = PathGenerator(
        waypoints=MULTI_PATH_WAYPOINTS_1,
        step_size=0.5,
        num_draws=1000,
        noise_scale=noise_scales,
    )
    gen.draw()
