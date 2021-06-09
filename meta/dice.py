import torch


def magic_box(x):
    """DiCE operation that saves computation graph inside tensor
    See ``Implementation of DiCE'' section in the DiCE Paper for details
    Args:
        x (tensor): Input tensor
    Returns:
        1 (tensor): Tensor that has computation graph saved
    References:
        https://github.com/alshedivat/lola/blob/master/lola_dice/rpg.py
        https://github.com/alexis-jacq/LOLA_DiCE/blob/master/ipd_DiCE.py
    """
    return torch.exp(x - x.detach())


def get_dice_loss(logprobs, reward, value, args, i_agent, is_train):
    """Compute DiCE loss
    In our code, we use DiCE in the inner loop to be able to keep the dependency in the
    adapted parameters. This is required in order to compute the opponent shaping term.
    Args:
        logprobs (list): Contains log probability of all agents
        reward (list): Contains rewards across trajectories for specific agent
        value (tensor): Contains value for advantage computed via linear baseline
        args (argparse): Python argparse that contains arguments
        i_agent (int): Agent to compute DiCE loss for
        is_train (bool): Flag to identify whether in meta-train or not
    Returns:
        dice loss (tensor): DiCE loss with baseline reduction
    References:
        https://github.com/alshedivat/lola/blob/master/lola_dice/rpg.py
        https://github.com/alexis-jacq/LOLA_DiCE/blob/master/ipd_DiCE.py
    """
    # Get discounted_reward
    reward = torch.stack(reward, dim=1)
    cum_discount = torch.cumprod(args.discount * torch.ones(*reward.size()), dim=1) / args.discount
    discounted_reward = reward * cum_discount

    # Compute stochastic nodes involved in reward dependencies
    if args.opponent_shaping and is_train:
        logprob_sum, stochastic_nodes = 0., 0.
        for logprob in logprobs:
            logprob = torch.stack(logprob, dim=1)
            logprob_sum += logprob
            stochastic_nodes += logprob
        dependencies = torch.cumsum(logprob_sum, dim=1)
    else:
        logprob = torch.stack(logprobs[i_agent], dim=1)
        dependencies = torch.cumsum(logprob, dim=1)
        stochastic_nodes = logprob

    # Get DiCE loss
    dice_loss = torch.mean(torch.sum(magic_box(dependencies) * discounted_reward, dim=1))

    # Apply variance_reduction if value is provided
    baseline_term = 0.
    if value is not None:
        discounted_value = value.detach() * cum_discount
        baseline_term = torch.mean(torch.sum((1 - magic_box(stochastic_nodes)) * discounted_value, dim=1))

    return -(dice_loss + baseline_term)
