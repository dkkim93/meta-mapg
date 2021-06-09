import torch
import gym
import numpy as np
import torch.nn.functional as F
from gym_env import make_env
from meta.dice import get_dice_loss
from meta.linear_baseline import LinearFeatureBaseline
from torch.autograd import Variable
from torch.distributions import Categorical, Normal
from collections import OrderedDict
from misc.torch_utils import get_parameters, get_named_parameters
from misc.rl_utils import get_return


class Base(object):
    """Base class that has shared methods between a meta-agent and an opponent

    Args:
        log (dict): Dictionary that contains python logging
        tb_writer (SummeryWriter): Used for tensorboard logging
        args (argparse): Python argparse that contains arguments
        name (str): Specifies agent's name
        i_agent (int): Agent index among the agents in the shared environment
        rank (int): Used for thread-specific meta-agent for multiprocessing. Default: -1
    """
    def __init__(self, log, tb_writer, args, name, i_agent, rank):
        super(Base, self).__init__()

        self.log = log
        self.tb_writer = tb_writer
        self.args = args
        self.name = name + str(i_agent)
        self.i_agent = i_agent
        self.rank = rank

    def _set_dim(self):
        env = make_env(self.args)
        if isinstance(env.observation_space[self.i_agent], gym.spaces.Box):
            self.input_dim = env.observation_space[self.i_agent].shape[0]
        else:
            self.input_dim = env.observation_space[self.i_agent].n
        if isinstance(env.action_space[self.i_agent], gym.spaces.Box):
            self.output_dim = env.action_space[self.i_agent].shape[0]
        else:
            self.output_dim = env.action_space[self.i_agent].n
        env.close()

        self.log[self.args.log_name].info("[{}] Input dim: {}".format(
            self.name, self.input_dim))
        self.log[self.args.log_name].info("[{}] Output dim: {}".format(
            self.name, self.output_dim))

    def _set_action_type(self):
        env = make_env(self.args)
        if isinstance(env.action_space[self.i_agent], gym.spaces.Discrete):
            self.is_discrete_action = True
            self.action_dtype = int
        else:
            self.is_discrete_action = False
            self.action_dtype = float
        env.close()

        self.log[self.args.log_name].info("[{}] Discrete action space: {}".format(
            self.name, self.is_discrete_action))

    def _set_linear_baseline(self):
        self.linear_baseline = LinearFeatureBaseline(
            input_size=self.input_dim, args=self.args)

    def _get_value_loss(self, value, reward):
        value = torch.stack(value, dim=1)
        return_ = get_return(reward, self.args)
        assert value.shape == return_.shape
        return F.mse_loss(value, return_)

    def reset_lstm_state(self):
        if hasattr(self, 'is_tabular_policy'):
            return

        self.actor_hidden = (
            Variable(torch.zeros(self.args.traj_batch_size, self.args.n_hidden)),
            Variable(torch.zeros(self.args.traj_batch_size, self.args.n_hidden)))

        if "meta" in self.name:
            self.value_hidden = (
                Variable(torch.zeros(self.args.traj_batch_size, self.args.n_hidden)),
                Variable(torch.zeros(self.args.traj_batch_size, self.args.n_hidden)))

    def act(self, obs, actor):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        # Compute action probability
        if hasattr(self, 'is_tabular_policy'):
            obs = torch.argmax(obs, dim=1)
            probs = F.softmax(actor[obs, :], dim=1)
        else:
            params = actor if isinstance(actor, OrderedDict) else None
            if self.is_discrete_action:
                probs, self.actor_hidden = self.actor((obs, self.actor_hidden), params=params)
                probs = probs.squeeze(-1)
            else:
                mu, scale, self.actor_hidden = self.actor((obs, self.actor_hidden), params=params)

        # Compute action, logprob, and entropy
        if self.is_discrete_action:
            distribution = Categorical(probs=probs)
        else:
            distribution = Normal(loc=mu, scale=scale)
        action = distribution.sample()
        logprob = distribution.log_prob(action)
        if len(logprob.shape) == 2:
            logprob = torch.sum(logprob, dim=-1)
        entropy = distribution.entropy()
        if len(entropy.shape) == 2:
            entropy = torch.sum(entropy, dim=-1)

        # Compute value for advantage at outer loop
        if "meta" in self.name:
            value, self.value_hidden = self.value((obs, self.value_hidden), params=None)
            value = value.squeeze(-1)
        else:
            value = None

        return action.numpy().astype(self.action_dtype), logprob, entropy, value

    def inner_update(self, actor, memory, i_joint, is_train):
        if i_joint == self.args.chain_horizon:
            return None

        obs, logprobs, _, _, rewards = memory.sample()

        # Compute value for baseline
        reward = rewards[self.i_agent]
        value = self.linear_baseline(obs, reward)

        # Compute DiCE loss
        actor_loss = get_dice_loss(logprobs, reward, value, self.args, self.i_agent, is_train)

        # Get adapted parameters
        actor_grad = torch.autograd.grad(actor_loss, get_parameters(actor), create_graph=is_train)

        if hasattr(self, 'is_tabular_policy'):
            phi = actor - 1. * actor_grad[0]
        else:
            phi = OrderedDict()
            lr = self.dynamic_lr[i_joint] if "meta" in self.name else self.args.actor_lr_inner
            for (name, param), grad in zip(get_named_parameters(actor), actor_grad):
                phi[name] = param - lr * grad

        return phi

    def _set_policy(self):
        raise NotImplementedError()

    def _set_dynamic_lr(self):
        raise NotImplementedError()

    def share_memory(self):
        raise NotImplementedError()

    def sync(self):
        raise NotImplementedError()

    def set_persona(self):
        raise NotImplementedError()

    def get_outer_loss(self, memories, iteration):
        raise NotImplementedError()
