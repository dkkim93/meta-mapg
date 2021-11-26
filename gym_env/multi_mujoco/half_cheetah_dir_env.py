import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class HalfCheetahDirEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Custom half cheetah environment that returns rewards based on
    either left (direction: -1) or right (direction: +1).

    Args:
        args (argparse): Python argparse that contains arguments

    References:
        https://github.com/openai/gym/blob/master/gym/envs/mujoco/half_cheetah.py
    """
    def __init__(self, args):
        self.args = args
        assert self.args.direction == -1 or self.args.direction == +1, \
            "Only left (direction = -1) and right (direction = +1) are supported"

        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        # Step forward simulation
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        # Get next obs
        next_obs = self.get_obs()

        # Compute reward based on direction
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = self.args.direction * (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run

        # Set done and info
        done, info = False, {}

        return next_obs, reward, done, info

    def get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self.get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
