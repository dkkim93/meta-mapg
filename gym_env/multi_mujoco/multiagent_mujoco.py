import gym
import random
import numpy as np
from gym_env.wrapper import NormalizedActionWrapper
from gym.spaces import Box


class MultiAgentMuJoCo(object):
    """Multiagent MuJoCo environment that converts single agent MuJoCo env's into
    multiagent settins
    
    Args:
        args (argparse): Python argparse that contains arguments

    References:
        https://github.com/schroederdewitt/multiagent_mujoco
    """
    def __init__(self, args):
        self.args = args

        # Set env
        self.env = gym.make(args.env_name, args=args)
        self.env._max_episode_steps = self.args.ep_horizon
        self.env = NormalizedActionWrapper(self.env)

        # Set observation space per agent. Assuming state observation assumption in Markov game, 
        # observation space is same as original env's observation space
        self.observation_space = [self.env.observation_space for _ in range(self.args.n_agent)]

        # Set action space per agent. Assuming all agents have same action space dimension, 
        # action space is original env's action space divided by number of agents 
        action_dim = int(self.env.action_space.shape[0] / args.n_agent)
        self.action_space = [
            Box(
                low=self.env.action_space.low[:action_dim], 
                high=self.env.action_space.high[:action_dim], 
                dtype=np.float32)
            for _ in range(self.args.n_agent)]

    def seed(self, value):
        random.seed(value)
        np.random.seed(value)
        self.env.seed(value)

    def reset(self):
        self.env.reset()
        return self.env.get_obs()

    def step(self, actions):
        actions = np.concatenate(actions)
        observations, reward, done, info = self.env.step(actions)
        if self.args.reward_scale:
            reward = reward / 100.
        rewards = [reward for _ in range(self.args.n_agent)]
        return observations, rewards, done, info

    def render(self, **kwargs):
        self.env.render(**kwargs)

    def close(self):
        self.env.close()

    @staticmethod
    def sample_personas(is_train, is_val=True, path="./"):
        path = path + "pretrain_model/HalfCheetahDir-v0/"

        if is_train:
            iterations = np.load(path + "/train.npy")
            iteration = np.random.choice(iterations)
            filepath = path + \
                "actor::1_" + \
                "iteration::" + str(iteration) + ".pth"
            return [{"iteration": iteration, "filepath": filepath}]
        else:
            personas = []
            iterations = np.load(path + "/val.npy") if is_val else np.load(path + "/test.npy")
            for iteration in iterations:
                filepath = path + \
                    "actor::1_" + \
                    "iteration::" + str(iteration) + ".pth"
                personas.append({"iteration": iteration, "filepath": filepath})
            return personas
