import torch


class ReplayMemory(object):
    """Simple replay memory that contains trajectories for each task 
    in a Markov chain
    Args:
        args (argparse): Python argparse that contains arguments
    """
    def __init__(self, args):
        self.args = args

        self.obs = []
        self.logprobs = [[] for _ in range(args.n_agent)]
        self.entropies = [[] for _ in range(args.n_agent)]
        self.values = [[] for _ in range(args.n_agent)]
        self.rewards = [[] for _ in range(args.n_agent)]

    def add(self, obs, logprobs, entropies, values, rewards):
        self.obs.append(obs)

        for logprob_memory, logprob in zip(self.logprobs, logprobs):
            logprob_memory.append(logprob)

        for entropy_memory, entropy in zip(self.entropies, entropies):
            entropy_memory.append(entropy)

        for value_memory, value in zip(self.values, values):
            value_memory.append(value)

        if isinstance(rewards, list):
            for reward_memory, reward in zip(self.rewards, rewards):
                reward_memory.append(torch.from_numpy(reward).float())
        else:
            for i_agent in range(self.args.n_agent):
                self.rewards[i_agent].append(torch.from_numpy(rewards[:, i_agent]).float())
        
    def sample(self):
        return self.obs, self.logprobs, self.entropies, self.values, self.rewards
