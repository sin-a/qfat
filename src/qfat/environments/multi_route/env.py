"""Adapted from https://github.com/notmahi/bet."""

import logging
from typing import List

import gym
import gym.spaces as spaces
import matplotlib.pyplot as plt
import numpy as np

from qfat.environments.multi_route import multi_route

logger = logging.getLogger(__name__)


class MultiRouteEnvV1(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}
    NUM_ENV_STEPS = 50

    def __init__(
        self,
        track_trajectories: bool = False,
        random_reset: bool = False,
    ) -> None:
        super().__init__()
        self.random_reset = random_reset
        self.obs_high_bound = 2 * multi_route.BOUNDS + 2
        self.obs_low_bound = -2
        self._dim = 2
        self._noise_scale = 0.25
        self._starting_point = np.zeros(self._dim)
        self.track_trajectories = track_trajectories
        action_limit = 2
        self._trajectory = []
        self.action_space = spaces.Box(
            -action_limit,
            action_limit,
            shape=self._starting_point.shape,
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(
            self.obs_low_bound,
            self.obs_high_bound,
            shape=self._starting_point.shape,
            dtype=np.float64,
        )

        self._target_bounds = spaces.Box(
            2 * multi_route.BOUNDS - 1,
            2 * multi_route.BOUNDS + 1,
            shape=self._starting_point.shape,
            dtype=np.float64,
        )
        self._fig, self._ax = plt.subplots()
        if self.track_trajectories:
            self._trajectories = []

    def reset(self):
        self._trajectory = []
        self._state = np.copy(self._starting_point)
        self._trajectory.append(np.copy(self._state))
        if self.random_reset:
            self._state += np.random.normal(
                0, self._noise_scale, size=self._starting_point.shape
            )
        plt.close("all")
        self._fig, self._ax = plt.subplots()
        return np.copy(self._state)

    @property
    def completed_tasks(self) -> List[str]:
        return [self.get_trajectory_label(self._trajectory)]

    def get_trajectory_label(self, trajectory) -> str:
        if len(trajectory) < 15:
            return "gray"
        point_init = trajectory[0]
        point_ref = trajectory[14]
        diff = point_ref - point_init
        if diff[0] > diff[1]:
            return "lightsteelblue"
        return "rosybrown"

    def step(self, action):
        if not self.action_space.contains(action):
            logger.warning(
                f"{action!r} ({type(action)}) invalid. Clipping action to bounds."
            )
            action = np.clip(
                action, a_min=self.action_space.low, a_max=self.action_space.high
            )

        self._state += action
        reward = 0
        done = False

        if self._target_bounds.contains(self._state):
            reward = 1
            done = True
        if not self.observation_space.contains(self._state):
            reward = -1
            done = True
        self._trajectory.append(np.copy(self._state))
        if done and len(self._trajectory) > 0 and self.track_trajectories:
            self._trajectories.append(self._trajectory)
        return np.copy(self._state), reward, done, {}

    def draw_trajectory(self, trajectory) -> None:
        color = self.get_trajectory_label(trajectory)
        for i in range(len(trajectory) - 1):
            point, next_point = trajectory[i], trajectory[i + 1]
            self._ax.plot(
                [point[0], next_point[0]],
                [point[1], next_point[1]],
                color=color,
                alpha=0.7,
                linewidth=2,
            )

    def render(self, mode="human"):
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
        else:
            self._ax.clear()

        self._ax.set_aspect("equal", "box")
        self._fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self._ax.axis("off")
        self._ax.set_facecolor("white")
        self._fig.patch.set_facecolor("white")

        obs_rectangle = plt.Rectangle(
            (self.obs_low_bound, self.obs_low_bound),
            self.obs_high_bound - self.obs_low_bound,
            self.obs_high_bound - self.obs_low_bound,
            fill=False,
            edgecolor="lightgray",
            linestyle="--",
            linewidth=1,
        )
        self._ax.add_patch(obs_rectangle)

        target_rectangle = plt.Rectangle(
            (self._target_bounds.low[0], self._target_bounds.low[1]),
            self._target_bounds.high[0] - self._target_bounds.low[0],
            self._target_bounds.high[1] - self._target_bounds.low[1],
            fill=True,
            color="#A1D99B",
            alpha=0.7,
        )
        self._ax.add_patch(target_rectangle)
        self.draw_trajectory(self._trajectory)
        if self.track_trajectories:
            for traj in self._trajectories:
                self.draw_trajectory(traj)
        self._ax.plot(
            self._state[0],
            self._state[1],
            marker="o",
            markersize=10,
            color="#FB6A4A",
            markeredgecolor="black",
            markeredgewidth=1,
        )

        padding = 1
        self._ax.set_xlim(self.obs_low_bound - padding, self.obs_high_bound + padding)
        self._ax.set_ylim(self.obs_low_bound - padding, self.obs_high_bound + padding)

        if mode == "human":
            plt.show(block=False)
            plt.pause(0.1)
        elif mode == "rgb_array":
            self._fig.canvas.draw()
            image = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            return image
        else:
            raise NotImplementedError(f"Render mode '{mode}' not implemented.")

    def set_state(self, state):
        err_msg = f"{state!r} ({type(state)}) invalid"
        assert self.observation_space.contains(state), err_msg
        self._state = np.copy(state)
        return self._state


if __name__ == "__main__":
    env = MultiRouteEnvV1()
    s = env.reset()
    done = False
    while not done:
        action = -4 * np.random.rand(2) + 2
        s, r, done, meta = env.step(action=action)
        env.render("human")
        print(f"REWARD: {r}")
