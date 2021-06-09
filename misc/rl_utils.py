import torch
import numpy as np
from misc.replay_memory import ReplayMemory


def collect_trajectory(agents, actors, env, args):
    """Collect batch of trajectories

    Args:
        agents (list): Contains agents that act in the environment
        actors (list): Contains parameters that agents use to select action
        env (gym): OpenAI Gym environment that agents execute actions
        args (argparse): Python argparse that contains arguments

    Returns:
        memory (ReplayMemory): Class that includes trajectories
        scores (list): Contains scores for each agent
    """
    # Initialize memory
    memory = ReplayMemory(args)

    # Initialize LSTM state
    for agent in agents:
        agent.reset_lstm_state()

    # Begin to collect trajectory
    obs = env.reset()
    scores = [0. for _ in range(args.n_agent)]

    for timestep in range(args.ep_horizon):
        # Get actions
        actions, logprobs, entropies, values = [], [], [], []
        for agent, actor in zip(agents, actors):
            action, logprob, entropy, value = agent.act(obs, actor)
            actions.append(action)
            logprobs.append(logprob)
            entropies.append(entropy)
            values.append(value)

        # Take step in the environment
        next_obs, rewards, _, _ = env.step(actions)

        # Add to memory
        memory.add(
            obs=obs,
            logprobs=logprobs,
            entropies=entropies,
            values=values,
            rewards=rewards)

        # Update scores
        for i_agent in range(args.n_agent):
            if isinstance(rewards, list):
                reward = np.mean(rewards[i_agent]) / float(args.ep_horizon)
            else:
                reward = np.mean(rewards[:, i_agent]) / float(args.ep_horizon)
            scores[i_agent] += reward

        # For next timestep
        obs = next_obs

    return memory, scores


def get_return(reward, args):
    """Compute episodic return given trajectory

    Args:
        reward (list): Contains rewards across trajectories for specific agent
        args (argparse): Python argparse that contains arguments

    Returns:
        return_ (torch.Tensor): Episodic return with shape: (batch, ep_horizon)
    """
    reward = torch.stack(reward, dim=1)
    assert reward.shape == (args.traj_batch_size, args.ep_horizon), \
        "Shape must be: (batch, ep_horizon)"

    R, return_ = 0., []
    for timestep in reversed(range(args.ep_horizon)):
        R = reward[:, timestep] + args.discount * R
        return_.insert(0, R)
    return_ = torch.stack(return_, dim=1)

    return return_


def get_entropy_loss(memory, args, i_agent):
    """Compute entropy loss for exploration

    Args:
        memory (ReplayMemory): Class that includes trajectories
        args (argparse): Python argparse that contains arguments
        i_agent (int): Index of agent to compute entropy loss

    Returns:
        entropy_loss (torch.Tensor): Entropy loss for encouraging exploration
    """
    _, _, entropies, _, _ = memory.sample()
    entropy = torch.stack(entropies[i_agent], dim=1)
    assert entropy.shape == (args.traj_batch_size, args.ep_horizon), \
        "Shape must be: (batch, ep_horizon)"

    entropy_loss = -args.entropy_weight * torch.mean(torch.sum(entropy, dim=1))
    return entropy_loss


def get_gae(value, reward, args, is_normalize=False, eps=1e-8):
    """Compute generalized advantage estimator

    Args:
        value (list): Contains value function across trajectories
        reward (list): Contains rewards across trajectories for specific agent
        args (argparse): Python argparse that contains arguments
        is_normalize (bool): Normalize baseline if flag is True. Default: False
        eps (float): Epsilon for numerical stability. Default: 1e-8

    Returns:
        GAE (torch.Tensor): Estimated generalized advantage function

    References:
        https://github.com/dgriff777/rl_a3c_pytorch/blob/master/train.py
    """
    value = torch.stack(value, dim=1)
    assert value.shape == (args.traj_batch_size, args.ep_horizon), \
        "Shape must be: (batch, ep_horizon)"
    value = torch.cat((value, torch.zeros(value.shape[0], 1)), dim=1)
    reward = torch.stack(reward, dim=1)
    assert reward.shape == (args.traj_batch_size, args.ep_horizon), \
        "Shape must be: (batch, ep_horizon)"

    gae, advantage = 0., []
    for timestep in reversed(range(args.ep_horizon)):
        delta = (reward[:, timestep] + args.discount * value[:, timestep + 1]) - value[:, timestep]
        gae = gae * args.discount * args.lambda_ + delta
        advantage.insert(0, gae)
    advantage = torch.stack(advantage, dim=1)
    assert reward.shape == advantage.shape

    if is_normalize:
        advantage = advantage - torch.mean(advantage)
        std = torch.sqrt(torch.mean(advantage ** 2))
        advantage.div_(std + eps)

    return advantage
