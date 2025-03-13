import os

import gymnasium
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env


class HalfCheetahFullObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 20}

    def __init__(self):
        asset_path = os.path.join(os.path.dirname(__file__), "half_cheetah.xml")
        mujoco_env.MujocoEnv.__init__(self, asset_path, 5, None, render_mode="human")
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate(
            [
                # self.data.qpos.flat[1:],
                self.data.qpos.flat,  # [1:],
                self.data.qvel.flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set(self, state):
        qpos_dim = self.data.qpos.size
        qpos = state[:qpos_dim]
        qvel = state[qpos_dim:]
        self.set_state(qpos, qvel)
        return self._get_obs()


class HalfCheetahWrapper(gymnasium.Wrapper):
    def __init__(self):
        super().__init__(HalfCheetahFullObsEnv())

    def render(self, mode=None):
        """
        Render the environment with the specified mode.
        """
        if mode is not None:
            self.env.render_mode = mode
        return self.env.render()


if __name__ == "__main__":
    env = HalfCheetahWrapper()
    env.reset()
    for _ in range(1000):
        env.step(env.action_space.sample())
        env.render("human")
    env.close()
