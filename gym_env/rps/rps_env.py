import pickle
import random
import numpy as np
from gym_env.rps.base import Base
from gym.spaces import Discrete, Tuple
from misc.utils import to_onehot


class RPSEnv(Base):
    """Base class for two agent iterated rock-paper-scissors (RPS)
    Possible actions for each agent are (R)ock, (P)aper, and (S)cissors

    Args:
        args (argparse): Python argparse that contains arguments
    """
    def __init__(self, args):
        super(RPSEnv, self).__init__(args)

        self.observation_space = [Discrete(10) for _ in range(2)]
        self.states = np.arange(start=1, stop=10, step=1, dtype=np.int32)
        self.action_space = Tuple([Discrete(3) for _ in range(2)])

        self._set_payoff_matrix()
        self._set_states_dict()

    def reset(self):
        obs = np.zeros(self.args.traj_batch_size, dtype=np.int32)
        obs = to_onehot(obs, dim=10)
        return obs

    def step(self, actions):
        state = self._action_to_state(actions)
        assert len(state.shape) == 1, "Shape should be (traj_batch_size,)"

        # Get observation
        obs = self.states[state]
        obs = to_onehot(obs, dim=10)

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
        path = path + "pretrain_model/RPS-v0/"

        if is_train:
            with open(path + "rock/train", "rb") as fp:
                rock_personas = pickle.load(fp)
            with open(path + "paper/train", "rb") as fp:
                paper_personas = pickle.load(fp)
            with open(path + "scissors/train", "rb") as fp:
                scissors_personas = pickle.load(fp)
            return random.choices(rock_personas + paper_personas + scissors_personas, k=1)
        else:
            if is_val:
                with open(path + "rock/val", "rb") as fp:
                    rock_personas = pickle.load(fp)
                with open(path + "paper/val", "rb") as fp:
                    paper_personas = pickle.load(fp)
                with open(path + "scissors/val", "rb") as fp:
                    scissors_personas = pickle.load(fp)
            else:
                with open(path + "rock/test", "rb") as fp:
                    rock_personas = pickle.load(fp)
                with open(path + "paper/test", "rb") as fp:
                    paper_personas = pickle.load(fp)
                with open(path + "scissors/test", "rb") as fp:
                    scissors_personas = pickle.load(fp)
            return rock_personas + paper_personas + scissors_personas

    @staticmethod
    def generate_personas(n_train=200, n_val=20, n_test=20):
        path = "./pretrain_model/RPS-v0/"

        for persona_type in ["rock", "paper", "scissors"]:
            personas = []
            for _ in range(n_train + n_val + n_test):
                if persona_type == "rock":
                    rock_prob = np.random.uniform(low=1. / 3., high=1., size=(10,))
                    paper_prob = (1. - rock_prob) / 2.
                    scissors_prob = (1. - rock_prob) / 2.
                    assert np.array_equal(
                        np.argmax(np.stack([rock_prob, paper_prob, scissors_prob], axis=1), axis=1), np.zeros((10,)))
                elif persona_type == "paper":
                    paper_prob = np.random.uniform(low=1. / 3., high=1., size=(10,))
                    scissors_prob = (1. - paper_prob) / 2.
                    rock_prob = (1. - paper_prob) / 2.
                    assert np.array_equal(
                        np.argmax(np.stack([paper_prob, scissors_prob, rock_prob], axis=1), axis=1), np.zeros((10,)))
                elif persona_type == "scissors":
                    scissors_prob = np.random.uniform(low=1. / 3., high=1., size=(10,))
                    rock_prob = (1. - scissors_prob) / 2.
                    paper_prob = (1. - scissors_prob) / 2.
                    assert np.array_equal(
                        np.argmax(np.stack([scissors_prob, rock_prob, paper_prob], axis=1), axis=1), np.zeros((10,)))
                else:
                    raise ValueError()

                assert np.sum(rock_prob + paper_prob + scissors_prob) == 10.
                assert np.allclose((rock_prob + paper_prob + scissors_prob), np.ones((10,)))

                persona = np.log(np.stack([rock_prob, paper_prob, scissors_prob], axis=1))
                personas.append(persona)

            with open(path + persona_type + "/train", "wb") as fp:
                pickle.dump(personas[:n_train], fp)

            with open(path + persona_type + "/val", "wb") as fp:
                pickle.dump(personas[n_train:n_train + n_val], fp)

            with open(path + persona_type + "/test", "wb") as fp:
                pickle.dump(personas[n_train + n_val:], fp)
