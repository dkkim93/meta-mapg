import pickle
import random
import numpy as np
from gym_env.ipd.base import Base
from gym.spaces import Discrete, Tuple
from misc.utils import to_onehot


class IPDEnv(Base):
    """Base class for two agent prisoner's dilemma game
    Possible actions for each agent are (C)ooperate and (D)efect

    Args:
        args (argparse): Python argparse that contains arguments

    References:
        https://github.com/alexis-jacq/LOLA_DiCE/blob/master/envs/prisoners_dilemma.py
    """
    def __init__(self, args):
        super(IPDEnv, self).__init__(args)

        self.observation_space = [Discrete(5) for _ in range(2)]
        self.states = np.arange(start=1, stop=5, step=1, dtype=np.int32)
        self.action_space = Tuple([Discrete(2) for _ in range(2)])

        self._set_payoff_matrix()
        self._set_states_dict()

    def reset(self):
        obs = np.zeros(self.args.traj_batch_size, dtype=np.int32)
        obs = to_onehot(obs, dim=5)
        return obs

    def step(self, actions):
        state = self._action_to_state(actions)
        assert len(state.shape) == 1, "Shape should be (traj_batch_size,)"

        # Get observation
        obs = self.states[state]
        obs = to_onehot(obs, dim=5)

        # Get reward
        rewards = []
        for i_agent in range(2):
            rewards.append(self.payoff_matrix[i_agent][state])

        # Get done
        done = False

        return obs, rewards, done, {}

    def render(self, mode='human', close=False):
        raise NotImplementedError()

    @staticmethod
    def sample_personas(is_train, is_val=True, path="./"):
        path = path + "pretrain_model/IPD-v0/"

        if is_train:
            with open(path + "defective/train", "rb") as fp:
                defective_personas = pickle.load(fp)
            with open(path + "cooperative/train", "rb") as fp:
                cooperative_personas = pickle.load(fp)
            return random.choices(defective_personas + cooperative_personas, k=1)
        else:
            if is_val:
                with open(path + "defective/val", "rb") as fp:
                    defective_personas = pickle.load(fp)
                with open(path + "cooperative/val", "rb") as fp:
                    cooperative_personas = pickle.load(fp)
            else:
                with open(path + "defective/test", "rb") as fp:
                    defective_personas = pickle.load(fp)
                with open(path + "cooperative/test", "rb") as fp:
                    cooperative_personas = pickle.load(fp)
            return defective_personas + cooperative_personas
