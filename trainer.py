import random
import torch
import numpy as np
from gym_env import make_env
from meta.meta_agent import MetaAgent
from meta.peer import Peer
from misc.rl_utils import collect_trajectory
from misc.utils import log_performance
from tensorboardX import SummaryWriter

torch.set_num_threads(1)


def meta_train(shared_meta_agent, process_dict, rank, log, args):
    # Initialize train_iteration
    train_iteration = 0

    # Set thread-specific tb_writer
    tb_writer = SummaryWriter('./log/tb_{0}/rank::{1}'.format(args.log_name, str(rank)))

    # Set thread-specific env
    env = make_env(args)

    # Set thread-specific seeds
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    env.seed(args.seed + rank)

    # Initialize thread-specific meta_agent
    meta_agent = MetaAgent(log, tb_writer, args, name="meta-agent", i_agent=0, rank=rank)

    # Initialize thread-specific peer
    peer = Peer(log, tb_writer, args, name="peer", i_agent=1, rank=rank)

    # Set agents
    agents = [meta_agent, peer] 

    while True:
        # Sync thread-specific meta-agent with shared meta-agent
        meta_agent.sync(shared_meta_agent)

        # Set peer's persona
        persona = env.sample_personas(is_train=True)[0]
        peer.set_persona(persona)

        # Accumulate actor and value losses for outer-loop optimization 
        # through processing until the end of Markov chain
        actor_losses, value_losses = [], []
        actors = [agent.actor for agent in agents]
        memories = []

        for i_joint in range(args.chain_horizon + 1):
            iteration = train_iteration * (args.chain_horizon + 1) + i_joint

            # Collect trajectory
            memory, scores = collect_trajectory(agents, actors, env, args)
            memories.append(memory)
            log_performance(scores, log, tb_writer, args, i_joint, train_iteration, rank, is_train=True)

            # Perform inner-loop update
            phis = []
            for agent, actor in zip(agents, actors):
                phi = agent.inner_update(actor, memory, i_joint, is_train=True)
                phis.append(phi)

            # Compute outer-loop loss
            actor_loss, value_loss = meta_agent.get_outer_loss(memories, iteration)
            actor_losses.append(actor_loss)
            value_losses.append(value_loss)

            # Clear unused variables in memory for saving RAM
            getattr(memory, "obs").clear()
            getattr(memory, "rewards").clear()

            # For next joint policy
            actors = phis

        # Perform outer update
        meta_agent.outer_update(shared_meta_agent, actor_losses, process_dict, train_iteration, update_type="actor")
        meta_agent.outer_update(shared_meta_agent, actor_losses, process_dict, train_iteration, update_type="dynamic_lr")
        meta_agent.outer_update(shared_meta_agent, value_losses, process_dict, train_iteration, update_type="value")

        # Update train_iteration
        train_iteration += 1
        process_dict[str(rank) + "/train_iteration"] = train_iteration

        # Terminate train
        if train_iteration >= args.max_train_iteration:
            import sys
            sys.exit()
