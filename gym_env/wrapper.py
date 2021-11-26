import gym
import numpy as np
from gym import spaces


class NormalizedActionWrapper(gym.ActionWrapper):
    """Environment wrapper to normalize the action space to [-scale, scale]

    Args:
        env (gym.env): OpenAI Gym environment to wrap around
        scale (float): Scale for normalizing action. Default: 1.0.

    References:
        https://github.com/tristandeleu/pytorch-maml-rl
    """
    def __init__(self, env, scale=1.0):
        super(NormalizedActionWrapper, self).__init__(env)

        self.scale = scale
        self.action_space = spaces.Box(low=-scale, high=scale, shape=self.env.action_space.shape)

    def action(self, action):
        # Clip the action in [-scale, scale]
        action = np.clip(action, -self.scale, self.scale)

        # Map normalized action to original action space
        lb, ub = self.env.action_space.low, self.env.action_space.high
        if np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)):
            action = lb + (action + self.scale) * (ub - lb) / (2 * self.scale)
            action = np.clip(action, lb, ub)
        else:
            raise ValueError("Invalid value in action space")

        return action
