import gym
import numpy as np


class Base(gym.Env):
    """Base class for two agent prisoner's dilemma game
    Possible actions for each agent are (C)ooperate and (D)efect

    Args:
        args (argparse): Python argparse that contains arguments

    References:
        https://github.com/alexis-jacq/LOLA_DiCE/blob/master/envs/prisoners_dilemma.py
    """
    def __init__(self, args):
        super(Base, self).__init__()
        assert args.n_agent == 2, "Only two agents are supported in this domain"

        self.args = args

    def _set_payoff_matrix(self):
        self.payoff_matrix = [
            np.array([-0.5, +1.5, -1.5, 0.5], dtype=np.float32),
            np.array([-0.5, -1.5, +1.5, 0.5], dtype=np.float32)]

    def _set_states_dict(self):
        self.states_dict = {}
        self.states_dict["S0"] = [np.array((0,), dtype=np.int64) for _ in range(2)]
        self.states_dict["DD"] = [np.array((1,), dtype=np.int64) for _ in range(2)]
        self.states_dict["DC"] = [np.array((2,), dtype=np.int64) for _ in range(2)]
        self.states_dict["CD"] = [np.array((3,), dtype=np.int64) for _ in range(2)]
        self.states_dict["CC"] = [np.array((4,), dtype=np.int64) for _ in range(2)]

    def _action_to_state(self, actions):
        assert actions[0].shape == actions[1].shape
        action0, action1 = actions
        state = 2 * action0 + action1
        return state
