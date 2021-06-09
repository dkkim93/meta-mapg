import gym
import numpy as np


class Base(gym.Env):
    """Base class for two agent iterated rock-paper-scissors (RPS)
    Possible actions for each agent are (R)ock, (P)aper, and (S)cissors

    Args:
        args (argparse): Python argparse that contains arguments
    """
    def __init__(self, args):
        super(Base, self).__init__()
        assert args.n_agent == 2, "Only two agents are supported in this domain"

        self.args = args

    def _set_payoff_matrix(self):
        self.payoff_matrix = [
            np.array([0., -1., +1., +1., 0., -1., -1., +1., 0.], dtype=np.float32),
            np.array([0., +1., -1., -1., 0., +1., +1., -1., 0.], dtype=np.float32)]

    def _set_states_dict(self):
        self.states_dict = {}
        self.states_dict["S0"] = [np.array((0,), dtype=np.int64) for _ in range(2)]
        self.states_dict["RR"] = [np.array((1,), dtype=np.int64) for _ in range(2)]
        self.states_dict["RP"] = [np.array((2,), dtype=np.int64) for _ in range(2)]
        self.states_dict["RS"] = [np.array((3,), dtype=np.int64) for _ in range(2)]
        self.states_dict["PR"] = [np.array((4,), dtype=np.int64) for _ in range(2)]
        self.states_dict["PP"] = [np.array((5,), dtype=np.int64) for _ in range(2)]
        self.states_dict["PS"] = [np.array((6,), dtype=np.int64) for _ in range(2)]
        self.states_dict["SR"] = [np.array((7,), dtype=np.int64) for _ in range(2)]
        self.states_dict["SP"] = [np.array((8,), dtype=np.int64) for _ in range(2)]
        self.states_dict["SS"] = [np.array((9,), dtype=np.int64) for _ in range(2)]

    def _action_to_state(self, actions):
        assert actions[0].shape == actions[1].shape
        action0, action1 = actions
        state = 3 * action0 + action1
        return state
