import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1
from gym import Wrapper

from qfat.constants import KITCHEN_DATA_PATH
from qfat.environments.env import ConditionalEnv
from qfat.normalizer.normalizer import MeanStdNormalizer

OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}
OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}
BONUS_THRESH = 0.3
ALL_TASKS = [
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "microwave",
    "kettle",
]


class KitchenEnv(KitchenTaskRelaxV1, ConditionalEnv):
    def __init__(
        self,
        start_from_data: bool = False,
        mask_goals: bool = True,
        q_init_path: Optional[str] = None,
        v_init_path: Optional[str] = None,
        tasks: Optional[List[str]] = None,
        complete_in_any_order: bool = True,
        terminate_on_task_complete: bool = True,
    ):
        self.mask_goals = mask_goals
        self.completed_tasks = []
        self._tasks = tasks or copy.copy(ALL_TASKS)
        self.tasks_to_complete = copy.copy(self._tasks)
        self.remove_task_when_completed = True
        self.terminate_on_task_complete = terminate_on_task_complete
        self.complete_in_any_order = complete_in_any_order

        super(KitchenEnv, self).__init__()
        self.start_from_data = start_from_data
        if start_from_data:
            q_init_path = q_init_path or str(KITCHEN_DATA_PATH / "all_init_qpos.npy")
            v_init_path = v_init_path or str(KITCHEN_DATA_PATH / "all_init_qvel.npy")
            self.init_qpos = np.load(q_init_path)
            self.init_qvel = np.load(v_init_path)

    def reset(self, **kwargs) -> np.ndarray:
        self.tasks_to_complete = copy.copy(self._tasks)
        self.completed_tasks = []
        if self.start_from_data:
            ind = np.random.randint(len(self.init_qpos))
            qpos, qvel = self.init_qpos[ind], self.init_qvel[ind]
            self.set_state(qpos, qvel)
            obs, _, _, _ = self.step(np.zeros(self.action_space.shape))
            return obs
        obs = super().reset()
        if self.mask_goals:
            obs = obs[:30]
        return obs

    def _get_task_goal(self):
        new_goal = np.zeros_like(self.goal)
        for element in self._tasks:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal
        return new_goal

    def reset_model(self):
        self.tasks_to_complete = copy.copy(self._tasks)
        return super().reset_model()

    @property
    def tasks(self):
        return self._tasks

    @tasks.setter
    def tasks(self, value: List[str]) -> None:
        self._tasks = copy(value)
        self.tasks_to_complete = copy.copy(self._tasks)

    def set_task_goal(self, indices: Optional[np.ndarray] = None) -> None:
        if indices is not None:
            self._tasks = [ALL_TASKS[i] for i in indices]
            print(f"Setting tasks to be achieved to: {self._tasks}")
            self.tasks_to_complete = copy.copy(self._tasks)

    def _get_reward_n_score(self, obs_dict):
        reward_dict, score = super()._get_reward_n_score(obs_dict)
        next_q_obs = obs_dict["qp"]
        next_obj_obs = obs_dict["obj_qp"]
        next_goal = self._get_task_goal()
        idx_offset = len(next_q_obs)
        completions = []
        all_completed_so_far = True
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] - next_goal[element_idx]
            )
            complete = distance < BONUS_THRESH
            if (
                complete and all_completed_so_far
                if not self.complete_in_any_order
                else complete
            ):
                completions.append(element)
                self.completed_tasks.append(element)
            all_completed_so_far = all_completed_so_far and complete
        if self.remove_task_when_completed:
            [self.tasks_to_complete.remove(element) for element in completions]
        bonus = float(len(completions))
        reward_dict["bonus"] = bonus
        reward_dict["r_total"] = bonus
        score = bonus
        return reward_dict, score

    def step(self, a, b=None):
        obs, reward, done, env_info = super().step(a, b=b)
        if self.terminate_on_task_complete:
            done = not self.tasks_to_complete
        if self.mask_goals:
            obs = obs[:30]
        return obs, reward, done, env_info

    def sample_tasks(
        self,
        n_tasks: int,
        as_str: bool = False,
    ) -> Union[List[int], List[str]]:
        if n_tasks > len(self._tasks):
            raise ValueError(
                "The number of tasks should not exceed the total number of tasks in the environment."
            )
        if as_str:
            tasks = np.array(self._tasks)
        else:
            tasks = np.arange(len(self._tasks))
        np.random.shuffle(tasks)
        return tasks[:n_tasks].tolist()

    def get_all_tasks(self, as_str: bool) -> Union[List[int], List[str]]:
        if not as_str:
            tasks = [i for i in range(len(self._tasks))]
        return tasks

    def get_task_names(self, labels: List[int]) -> List[str]:
        return [self._tasks[i] for i in labels]


class NormalizedKitchenWrapper(Wrapper):
    """
    Wraps the KitchenEnv environment to normalize observations (e.g., first part of obs)
    and unnormalize actions before stepping the environment.
    """

    def __init__(
        self,
        stats_path: str = str(KITCHEN_DATA_PATH / "data_stats.json"),
        normalize_actions: bool = True,
        normalize_states: bool = True,
        **kitchen_env_kwargs,
    ):
        """
        Args:
            stats_path (str): Path to the JSON file containing the saved normalization statistics.
            normalize_actions (bool): Whether to apply unnormalization to actions.
            normalize_states (bool): Whether to normalize states in obs.
            **kitchen_env_kwargs: Any additional kwargs passed to KitchenEnv constructor.
        """
        super().__init__(KitchenEnv(**kitchen_env_kwargs))
        self.normalizer = MeanStdNormalizer()
        self.normalizer.load_stats(Path(stats_path))

        self.normalize_actions = normalize_actions
        self.normalize_states = normalize_states

    def step(self, action):
        if self.normalize_actions:
            unnormalized_action = self.normalizer.unnormalize_action(action)
        else:
            unnormalized_action = action

        obs, reward, done, info = self.env.step(unnormalized_action.squeeze())
        if self.normalize_states:
            obs = np.array(obs, dtype=np.float32)
            obs = self.normalizer.normalize_state(obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if self.normalize_states:
            obs = np.array(obs, dtype=np.float32)
            obs = self.normalizer.normalize_state(obs)
        return obs


class ImageKitchenWrapper(Wrapper):
    def __init__(self, **kwargs):
        """
        Wraps the Kitchen environment to generate image states.

        Args:
            stats_path (str): Path to the JSON file containing the saved normalization statistics.
        """
        super().__init__(KitchenEnv(**kwargs))

    def resize_img(self, img):
        img = cv2.resize(
            img,
            (224, 224),
            interpolation=cv2.INTER_AREA,
        )
        return img

    def step(self, action) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        _, reward, done, info = self.env.step(action.squeeze())
        obs = self.resize_img(self.env.render("rgb_array"))
        return torch.from_numpy(obs.copy()), reward, done, info

    def reset(self, **kwargs) -> torch.Tensor:
        self.env.reset(**kwargs)
        obs = self.resize_img(self.env.render("rgb_array"))
        return torch.from_numpy(obs.copy())
