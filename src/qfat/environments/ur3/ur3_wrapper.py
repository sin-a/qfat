import logging
from collections import OrderedDict
from pathlib import Path

import gym
import gym_custom
import numpy as np
from gym_custom import spaces
from gym_custom.envs.custom.ur_utils import NullObjectiveBase
from gym_custom.envs.custom.ur_utils import URScriptWrapper_SingleUR3 as URScriptWrapper

from qfat.constants import UR3_DATA_PATH
from qfat.normalizer.normalizer import MinMaxNormalizer

logger = logging.getLogger(__name__)


def ur3_mask_robots():
    def transform(input):
        assert len(input) == 3, "Input length must be 4: (obs, act, goal)"
        # assume obs, act, mask, goal
        goal = input[-1].clone()
        goal[..., [0, 1]] = 0
        return (*input[:-1], goal)

    return transform


class UprightConstraint(NullObjectiveBase):
    def __init__(self):
        pass

    def _evaluate(self, SO3):
        axis_des = np.array([0, 0, -1])
        axis_curr = SO3[:, 2]
        return 1.0 - np.dot(axis_curr, axis_des)


class UR3Wrapper(gym.Wrapper):
    def __init__(
        self, stats_path: str = str(UR3_DATA_PATH / "data_stats.json"), goal_cond=False
    ):
        super(UR3Wrapper, self).__init__(
            gym_custom.make("single-ur3-xy-left-comb-larr-for-train-v0")
        )
        self.normalizer = MinMaxNormalizer()
        self.normalizer.load_stats(Path(stats_path))

        self.servoj_args, self.speedj_args = (
            {"t": None, "wait": None},
            {"a": 5, "t": None, "wait": None},
        )
        self.PID_gains = {
            "servoj": {"P": 1.0, "I": 0.5, "D": 0.2},
            "speedj": {"P": 0.20, "I": 10.0},
        }
        self.ur3_scale_factor = np.array([5, 5, 5, 5, 5, 5])
        self.gripper_scale_factor = np.array([1.0])
        self.env = URScriptWrapper(
            self.env, self.PID_gains, self.ur3_scale_factor, self.gripper_scale_factor
        )
        self.max_episode_steps = 1000
        # For action bound
        self.command_limits = {
            "movej": [np.array([-0.04, -0.04, 0]), np.array([0.04, 0.04, 0])]  # [m]
        }
        self.action_space = self._set_action_space()["movej"]
        # Set motor gain scale
        self.env.wrapper_right.ur3_scale_factor[:6] = [
            24.52907494,
            24.02851783,
            25.56517597,
            14.51868608,
            23.78797503,
            21.61325463,
        ]
        self.null_obj_func = UprightConstraint()
        self.state = None
        self.id = id
        self.absolute_pos = True
        self.completed_tasks = []
        self.goal_1 = np.array([0.0, -0.25])
        self.goal_2 = np.array([0.0, -0.40])
        self.goal_cond = goal_cond

    def convert_action_to_space(self, action_limits):
        if isinstance(action_limits, dict):
            space = spaces.Dict(
                OrderedDict(
                    [
                        (key, self.convert_action_to_space(value))
                        for key, value in self.command_limits.items()
                    ]
                )
            )
        elif isinstance(action_limits, list):
            low = action_limits[0]
            high = action_limits[1]
            space = gym_custom.spaces.Box(low, high, dtype=action_limits[0].dtype)
        else:
            raise NotImplementedError(type(action_limits), action_limits)
        return space

    def _set_action_space(self):
        return self.convert_action_to_space({"_": self.command_limits})

    def set_task_goal(self, task_goal):
        self.goal_1 = task_goal[2:4]
        self.goal_2 = task_goal[4:]

    def reset(self, *args, **kwargs):
        self.env.reset(*args, **kwargs)
        self.done = False
        self.goal1_achieved = False
        self.goal2_achieved = False
        self.episode_steps = 0
        self.dt = 1
        self.state = np.array([0.45, -0.325, 0.3, -0.25, 0.3, -0.40])
        self.completed_tasks = []
        self.goal_1 = np.array([0.0, -0.25])
        self.goal_2 = np.array([0.0, -0.40])
        return self.normalizer.normalize_state(self.state.copy())

    def step(self, action):
        if len(action) == 3:
            logger.info("Overwriting end effector height to 0.8")
            action = action[:2]
        unnormalized_action = self.normalizer.unnormalize_action(action)
        action = np.concatenate([unnormalized_action.squeeze(), [0.8]])
        q_right_des, _, _, _ = self.env.inverse_kinematics_ee(
            action, self.null_obj_func, arm="right"
        )
        qvel_right = (q_right_des - self.env.get_obs_dict()["right"]["qpos"]) / self.dt

        next_state, _, done, _ = self.env.step(
            {
                "right": {
                    "speedj": {
                        "qd": qvel_right,
                        "a": self.speedj_args["a"],
                        "t": self.speedj_args["t"],
                        "wait": self.speedj_args["wait"],
                    },
                    "move_gripper_force": {"gf": np.array([15.0])},
                }
            }
        )

        self.episode_steps += 1

        reward = 0
        if not self.goal1_achieved:
            self.goal1_achieved = self.check_goal1_achieved(next_state)
            if self.goal1_achieved:
                reward = 1
                self.completed_tasks.append("1")
        if not self.goal2_achieved:
            self.goal2_achieved = self.check_goal2_achieved(next_state)
            if self.goal2_achieved:
                reward = 1
                self.completed_tasks.append("2")
        done = (self.episode_steps >= self.max_episode_steps) or (
            self.goal1_achieved and self.goal2_achieved
        )
        self.state = next_state[:6]

        info = {}
        if self.goal_cond:
            if done:
                reward = self.calc_reward(next_state)
            else:
                reward = 0
        info["all_completions_ids"] = self.completed_tasks
        return self.normalizer.normalize_state(self.state.copy()), reward, done, info

    def calc_reward(self, next_state):
        block1_dist = np.linalg.norm(self.goal_1 - next_state[2:4], ord=1)
        block2_dist = np.linalg.norm(self.goal_2 - next_state[4:6], ord=1)
        return -(block1_dist + block2_dist)

    def check_goal1_achieved(self, next_state):
        if np.linalg.norm(self.goal_1 - next_state[2:4]) < 0.05:
            print("goal 1 achieved!")
            return True
        else:
            return False

    def check_goal2_achieved(self, next_state):
        if np.linalg.norm(self.goal_2 - next_state[4:6]) < 0.05:
            print("goal 2 achieved!")
            return True
        else:
            return False

    def __getattr__(self, attrname):
        return getattr(self.env, attrname)


if __name__ == "__main__":
    env = UR3Wrapper()
    done = False
    s = env.reset()
    while not done:
        s, r, done, *_ = env.step(env.action_space.sample())
        frame = env.render("human")
    env.close()
