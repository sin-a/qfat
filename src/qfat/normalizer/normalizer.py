import abc
import json
from typing import Optional

import numpy as np


class Normalizer(abc.ABC):
    def __init__(self, normalize_actions_flag=True):
        """
        Abstract class for Normalizer. Initializes statistics placeholders.

        Args:
            normalize_actions_flag (bool): Whether to normalize actions.
        """
        self.state_stats = {}
        self.action_stats = {}
        self.goal_stats = {}
        self.normalize_actions_flag = normalize_actions_flag

    @abc.abstractmethod
    def update_stats(self, states, actions, goals):
        pass

    @abc.abstractmethod
    def normalize_state(self, state):
        pass

    @abc.abstractmethod
    def normalize_action(self, action):
        pass

    @abc.abstractmethod
    def normalize_goal(self, goal):
        pass

    @abc.abstractmethod
    def unnormalize_state(self, normalized_state):
        pass

    @abc.abstractmethod
    def unnormalize_action(self, normalized_action):
        pass

    @abc.abstractmethod
    def unnormalize_goal(self, normalized_goal):
        pass

    def save_stats(self, path):
        stats = {
            "state_stats": self.state_stats,
            "action_stats": self.action_stats,
            "goal_stats": self.goal_stats,
        }
        with open(path, "w") as f:
            json.dump(stats, f)

    def load_stats(self, path):
        with open(path, "r") as f:
            stats = json.load(f)
            self.state_stats = stats["state_stats"]
            self.action_stats = stats["action_stats"]
            self.goal_stats = stats["goal_stats"]


class IdentityNormalizer(Normalizer):
    def update_stats(
        self,
        states: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        goals: Optional[np.ndarray] = None,
    ):
        self.state_stats = {}
        self.action_stats = {}
        self.goal_stats = {}

    def normalize_state(self, state):
        return state

    def normalize_action(self, action):
        if not self.normalize_actions_flag:
            return action
        return action

    def normalize_goal(self, goal):
        return goal

    def unnormalize_state(self, normalized_state):
        return normalized_state

    def unnormalize_action(self, normalized_action):
        if not self.normalize_actions_flag:
            return normalized_action
        return normalized_action

    def unnormalize_goal(self, normalized_goal):
        return normalized_goal


class MeanStdNormalizer(Normalizer):
    def update_stats(
        self,
        states: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        goals: Optional[np.ndarray] = None,
    ):
        if states is not None:
            state_ax = tuple(range(states.ndim - 1))
            state_mean = states.mean(axis=state_ax)
            state_std = states.std(axis=state_ax) + 1e-8
            self.state_stats = {"mean": state_mean.tolist(), "std": state_std.tolist()}

        if actions is not None:
            action_ax = tuple(range(actions.ndim - 1))
            action_mean = actions.mean(axis=action_ax)
            action_std = actions.std(axis=action_ax) + 1e-8
            self.action_stats = {
                "mean": action_mean.tolist(),
                "std": action_std.tolist(),
            }
        if goals is not None:
            goal_ax = tuple(range(goals.ndim - 1))
            goal_mean = goals.mean(axis=goal_ax)
            goal_std = goals.std(axis=goal_ax) + 1e-8
            self.goal_stats = {"mean": goal_mean.tolist(), "std": goal_std.tolist()}

    def normalize_state(self, state):
        mean = np.array(self.state_stats["mean"])
        std = np.array(self.state_stats["std"])
        return (state - mean) / std

    def normalize_action(self, action):
        if not self.normalize_actions_flag:
            return action
        mean = np.array(self.action_stats["mean"])
        std = np.array(self.action_stats["std"])
        return (action - mean) / std

    def normalize_goal(self, goal):
        mean = np.array(self.goal_stats["mean"])
        std = np.array(self.goal_stats["std"])
        return (goal - mean) / std

    def unnormalize_state(self, normalized_state):
        mean = np.array(self.state_stats["mean"])
        std = np.array(self.state_stats["std"])
        return normalized_state * std + mean

    def unnormalize_action(self, normalized_action):
        if not self.normalize_actions_flag:
            return normalized_action
        mean = np.array(self.action_stats["mean"])
        std = np.array(self.action_stats["std"])
        return normalized_action * std + mean

    def unnormalize_goal(self, normalized_goal):
        mean = np.array(self.goal_stats["mean"])
        std = np.array(self.goal_stats["std"])
        return normalized_goal * std + mean


class MinMaxNormalizer(Normalizer):
    def update_stats(
        self,
        states: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        goals: Optional[np.ndarray] = None,
    ):
        if states is not None:
            state_min = states.min(axis=0)
            state_max = states.max(axis=0)
            self.state_stats = {"min": state_min.tolist(), "max": state_max.tolist()}

        if actions is not None:
            action_min = actions.min(axis=0)
            action_max = actions.max(axis=0)
            self.action_stats = {"min": action_min.tolist(), "max": action_max.tolist()}

        if goals is not None:
            goal_min = goals.min(axis=0)
            goal_max = goals.max(axis=0)
            self.goal_stats = {"min": goal_min.tolist(), "max": goal_max.tolist()}

    def normalize_state(self, state):
        min_val = np.array(self.state_stats["min"])
        max_val = np.array(self.state_stats["max"])
        return 2 * (state - min_val) / (max_val - min_val + 1e-8) - 1

    def normalize_action(self, action):
        if not self.normalize_actions_flag:
            return action
        min_val = np.array(self.action_stats["min"])
        max_val = np.array(self.action_stats["max"])
        return 2 * (action - min_val) / (max_val - min_val + 1e-8) - 1

    def normalize_goal(self, goal):
        min_val = np.array(self.goal_stats["min"])
        max_val = np.array(self.goal_stats["max"])
        return 2 * (goal - min_val) / (max_val - min_val + 1e-8) - 1

    def unnormalize_state(self, normalized_state):
        if not self.state_stats:
            return normalized_state
        min_val = np.array(self.state_stats["min"])
        max_val = np.array(self.state_stats["max"])
        return (normalized_state + 1) * (max_val - min_val + 1e-8) / 2 + min_val

    def unnormalize_action(self, normalized_action):
        if not self.normalize_actions_flag:
            return normalized_action
        min_val = np.array(self.action_stats["min"])
        max_val = np.array(self.action_stats["max"])
        return (normalized_action + 1) * (max_val - min_val + 1e-8) / 2 + min_val

    def unnormalize_goal(self, normalized_goal):
        if not self.goal_stats:
            return normalized_goal
        min_val = np.array(self.goal_stats["min"])
        max_val = np.array(self.goal_stats["max"])
        return (normalized_goal + 1) * (max_val - min_val + 1e-8) / 2 + min_val
