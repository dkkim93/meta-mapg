import random
import numpy as np


def pc_grad(process_dict, rank, args, key, eps=1e-8):
    """Projection of conflicting gradients
    Args:
        process_dict (mp.Manger().dict()): Dictionary that contains gradients computed from each processor
        rank (int): Used for thread-specific meta-agent for multiprocessing. Default: -1
        args (argparse): Python argparse that contains arguments
        eps (float): Epsilon for numerical stability

    Returns:
        grad_i (np.ndarray): Projected gradient

    References:
        https://github.com/tianheyu927/PCGrad/blob/master/PCGrad_tf.py
    """

    key_i = str(rank) + key
    grad_i = np.copy(process_dict[key_i])
    for j_task in random.sample(range(args.n_process), args.n_process):
        key_j = str(j_task) + key
        if key_j in process_dict.keys():
            grad_j = process_dict[key_j]
            inner_product = np.sum(grad_i * grad_j)
            proj_direction = inner_product / (np.sum(grad_j * grad_j) + eps)
            grad_i = grad_i - min(proj_direction, 0.) * grad_j
    return grad_i
