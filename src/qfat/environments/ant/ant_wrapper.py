import warnings

from qfat.normalizer.normalizer import MinMaxNormalizer

warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path

import gym
import numpy as np

from qfat.constants import ANT_DATA_PATH
from qfat.environments.ant.ant_maze_multimodal import (
    AntMazeMultimodalEvalEnv,
)


class AntWrapper(gym.Wrapper):
    def __init__(
        self,
        goal_cond: bool = False,
        stats_path: str = str(ANT_DATA_PATH / "data_stats.json"),
    ):
        """
        Wraps the AntMazeMultimodalEvalEnv environment to normalize observations and unnormalize actions.

        Args:
            goal_cond (bool): Whether the environment is goal-conditioned.
            stats_path (str): Path to the JSON file containing saved normalization statistics.
        """
        super(AntWrapper, self).__init__(
            AntMazeMultimodalEvalEnv(render_mode="rgb_array")
        )

        # Load normalization stats
        self.normalizer = MinMaxNormalizer(normalize_actions_flag=True)
        self.normalizer.load_stats(Path(stats_path))

        # Set goal-conditioned behavior
        if goal_cond:
            self.env.set_goalcond()

    def reset(self, *args, **kwargs):
        """
        Reset the environment, normalize the observation, and handle goal-conditioning if applicable.
        """
        obs, *_ = self.env.reset(*args, **kwargs)
        self.num_achieved = 0
        self.env.completed_tasks = []
        # Goal-conditioned setup
        if self.env.goal_cond:
            one_indices = np.random.choice(4, 2, replace=False)
            self.env.set_achieved(one_indices)
            self.num_achieved = 2

        return_obs = np.concatenate((obs["observation"], obs["goal_arr"]))
        return_obs[29:37] = 0

        return return_obs

    def step(self, action):
        """
        Take a step in the environment, unnormalize the action, normalize the observation, and handle rewards.
        """
        unnormalized_action = self.normalizer.unnormalize_action(action)

        obs, reward, done, info = self.env.step(unnormalized_action)

        if self.num_achieved < np.sum(self.env.achieved):
            reward = 1
            self.num_achieved = np.sum(self.env.achieved)
        else:
            reward = 0

        return_obs = np.concatenate((obs["observation"], obs["goal_arr"]))
        return_obs[29:37] = 0
        return return_obs, reward, done, info

    def render(self, mode=None):
        """
        Render the environment with the specified mode.
        """
        if mode is not None:
            self.env.render_mode = mode
        return self.env.render()


if __name__ == "__main__":
    env = AntWrapper()
    s = env.reset()
    done = False
    i = 0
    while not done:
        s, r, done, *_ = env.step(env.action_space.sample())
        env.render("rgb_array")
    env.close()
