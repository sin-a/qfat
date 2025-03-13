from typing import Any, Tuple

import gym
import numpy as np

from qfat.datasets.dataset import Trajectory, TrajectoryDataset


class GoalAppendingWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        dataset: TrajectoryDataset,
        min_future_sep: int,
        future_seq_len: int,
        only_sample_tail: bool,
        use_labels_env_task: bool = False,
        use_raw_goals_env_task: bool = False,
        random_clip_trajectory: bool = False,
    ):
        super().__init__(env)
        self.dataset = dataset
        self.step_counter = 0
        self.ds_traj: Trajectory = None  # set when reset is called
        self.min_future_sep = min_future_sep
        self.future_seq_len = future_seq_len
        self.only_sample_tail = only_sample_tail
        self.random_clip_trajectory = random_clip_trajectory

        if use_labels_env_task is True and use_raw_goals_env_task is True:
            raise ValueError(
                "Should use either raw goals or labels for env goal setting, not both."
            )
        self.use_raw_goals_env_task = use_raw_goals_env_task
        self.use_labels_env_task = use_labels_env_task
        self.current_goal = None
        self.current_raw_goal = None
        self.rand_goal_idx = None

    def get_goals(self):
        valid_start_range = (
            self.step_counter + self.min_future_sep,
            len(self.ds_traj) - self.future_seq_len,
        )
        if valid_start_range[0] < valid_start_range[1]:
            if self.only_sample_tail:
                goals = self.ds_traj.goals[-self.future_seq_len :]
                raw_goals = (
                    self.ds_traj.goals[-self.future_seq_len :]
                    if self.ds_traj.raw_goals is not None
                    else None
                )
            else:
                future_start = np.random.randint(*valid_start_range)
                future_end = future_start + self.future_seq_len
                goals = self.ds_traj.goals[future_start:future_end]
                raw_goals = (
                    self.ds_traj.raw_goals[future_start:future_end]
                    if self.ds_traj.raw_goals is not None
                    else None
                )
        else:
            goals = self.ds_traj.goals[-self.future_seq_len :]
            raw_goals = (
                self.ds_traj.raw_goals[-self.future_seq_len :]
                if self.ds_traj.raw_goals is not None
                else None
            )
        return goals[None, ...], raw_goals

    def _get_goal_source(self) -> np.ndarray:
        """
        Determine the source for setting the environment's goal.

        Returns:
            np.ndarray: The goal to set, either from `labels` or `goals`.
        """
        goal = None
        if self.use_labels_env_task:
            timestep_task = self.ds_traj.labels.argmax(axis=1)
            labels, idx = np.unique(timestep_task, return_index=True)
            goal = labels[np.argsort(idx)] if self.step_counter == 0 else None
        elif self.use_raw_goals_env_task:
            goal = self.current_raw_goal.squeeze()
        else:
            goal = self.ds_traj.goals[-1][None,]  # self.current_goal.squeeze()
        return goal

    def set_env_goal(self) -> None:
        """
        Sets the goal in the environment based on the current trajectory and configuration.
        """
        env_goal = self._get_goal_source()
        if not self.use_labels_env_task and not self.use_raw_goals_env_task:
            env_goal = self.dataset.normalizer.unnormalize_goal(env_goal[-1])
        if self.use_raw_goals_env_task:
            env_goal = self.current_raw_goal[-1]
        self.env.set_task_goal(env_goal)

    def reset(self, **kwargs) -> Any:
        self.step_counter = 0
        self.ds_traj = self.dataset[np.random.choice(len(self.dataset))]
        if self.random_clip_trajectory:
            self.ds_traj = self.ds_traj[
                : np.random.randint(self.min_future_sep, len(self.ds_traj))
            ]
        obs = super().reset(**kwargs)
        self.current_goal, self.current_raw_goal = self.get_goals()
        self.set_env_goal()
        return obs

    def step(self, action: Any) -> Tuple:
        """Perform a step in the environment and set environment goal."""
        observation, reward, done, info = self.env.step(action)

        self.step_counter += 1
        self.current_goal, self.current_raw_goal = self.get_goals()
        self.set_env_goal()
        return observation, reward, done, info
