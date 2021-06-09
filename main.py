import time
import os
import argparse
import random
import torch
import torch.multiprocessing as mp
import numpy as np
from trainer import meta_train
from validator import meta_val
from tester import meta_test
from misc.utils import load_config, set_log
from meta.meta_agent import MetaAgent
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # Set logging
    if not os.path.exists("./log"):
        os.makedirs("./log")

    log = set_log(args)
    tb_writer = SummaryWriter('./log/tb_{0}'.format(args.log_name))

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed(args.seed)

    # For GPU, Set start method for multithreading
    if device == torch.device("cuda"):
        torch.multiprocessing.set_start_method('spawn')

    # Initialize shared meta-agent
    shared_meta_agent = MetaAgent(log, tb_writer, args, name="meta-agent", i_agent=0)
    shared_meta_agent.share_memory()

    # Begin either meta-train or meta-test
    if not args.test_mode:
        # Start meta-train
        processes, process_dict = [], mp.Manager().dict()
        for rank in range(args.n_process):
            p = mp.Process(
                target=meta_train,
                args=(shared_meta_agent, process_dict, rank, log, args))
            p.start()
            processes.append(p)
            time.sleep(0.1)

        p = mp.Process(
            target=meta_val,
            args=(shared_meta_agent, process_dict, -1, log, args))
        p.start()
        processes.append(p)
        time.sleep(0.1)

        for p in processes:
            time.sleep(0.1)
            p.join()
    else:
        # Start meta-test
        meta_test(shared_meta_agent, log, tb_writer, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="meta-mapg")

    # Algorithm
    parser.add_argument(
        "--opponent-shaping", action="store_true",
        help="If True, include opponent shaping in meta-optimization")
    parser.add_argument(
        "--traj-batch-size", type=int, default=64,
        help="Number of trajectories for each inner-loop update (K Hyperparameter)")
    parser.add_argument(
        "--n-process", type=int, default=5,
        help="Number of parallel processes for meta-optimization")
    parser.add_argument(
        "--actor-lr-inner", type=float, default=0.1,
        help="Learning rate for actor (inner loop)")
    parser.add_argument(
        "--actor-lr-outer", type=float, default=0.0001,
        help="Learning rate for actor (outer loop)")
    parser.add_argument(
        "--value-lr", type=float, default=0.00015,
        help="Learning rate for value (outer loop)")
    parser.add_argument(
        "--entropy-weight", type=float, default=0.5,
        help="Entropy weight in the meta-optimization")
    parser.add_argument(
        "--discount", type=float, default=0.96,
        help="Discount factor in reinforcement learning")
    parser.add_argument(
        "--lambda_", type=float, default=0.95,
        help="Lambda factor in GAE computation")
    parser.add_argument(
        "--chain-horizon", type=int, default=5,
        help="Markov chain terminates when chain horizon is reached")
    parser.add_argument(
        "--n-hidden", type=int, default=64,
        help="Number of neurons for hidden network")
    parser.add_argument(
        "--max-grad-clip", type=float, default=10.0,
        help="Max norm gradient clipping value in meta-optimization")
    parser.add_argument(
        "--test-mode", action="store_true",
        help="If True, perform meta-test instead of meta-train")
    parser.add_argument(
        "--max-train-iteration", type=int, default=1e5,
        help="Terminate program when max train iteration is reached")

    # Env
    parser.add_argument(
        "--env-name", type=str, default="",
        help="OpenAI gym environment name")
    parser.add_argument(
        "--ep-horizon", type=int, default=150,
        help="Episode is terminated when max timestep is reached")
    parser.add_argument(
        "--n-agent", type=int, default=2,
        help="Number of agents in a shared environment")

    # Misc
    parser.add_argument(
        "--seed", type=int, default=1,
        help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument(
        "--prefix", type=str, default="",
        help="Prefix for tb_writer and logging")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config that replaces default params with experiment specific params")

    args = parser.parse_args()

    # Load experiment specific config if provided
    if args.config is not None:
        load_config(args)

    # Set log name
    args.log_name = \
        "env::%s_seed::%s_opponent_shaping::%s_traj_batch_size::%s_chain_horizon::%s_" \
        "actor_lr_inner::%s_actor_lr_outer::%s_value_lr::%s_entropy_weight::%s_" \
        "max_grad_clip::%s_prefix::%s_log" % (
            args.env_name, args.seed, args.opponent_shaping, args.traj_batch_size, args.chain_horizon,
            args.actor_lr_inner, args.actor_lr_outer, args.value_lr, args.entropy_weight,
            args.max_grad_clip, args.prefix)

    main(args=args)
