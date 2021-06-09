import random
import torch
import numpy as np
from meta.meta_agent import MetaAgent
from meta.peer import Peer
from misc.rl_utils import collect_trajectory
from misc.utils import log_performance
from gym_env import make_env
from tensorboardX import SummaryWriter

torch.set_num_threads(1)


def meta_val(shared_meta_agent, process_dict, rank, log, args):
    # Initialize val_iteration and best_score
    val_iteration, best_score = 0, -np.inf

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

        # Get meta-validation persona
        personas = env.sample_personas(is_train=False, is_val=True)

        # Agents adapt until the end of Markov chain
        score = 0.
        for i_persona, persona in enumerate(personas): 
            peer.set_persona(persona)
            actors = [agent.actor for agent in agents]

            for i_joint in range(args.chain_horizon + 1):
                # Collect trajectory
                memory, scores = collect_trajectory(agents, actors, env, args)
                log_performance(scores, log, tb_writer, args, i_joint, val_iteration, rank, is_train=False)

                # Add score for validation
                if i_joint > 0:
                    score += scores[0]

                # Perform inner-loop update
                phis = []
                for agent, actor in zip(agents, actors):
                    phi = agent.inner_update(actor, memory, i_joint, is_train=False)
                    phis.append(phi)

                # For next joint policy
                actors = phis

        val_iteration += 1
        tb_writer.add_scalars("val_score", {"score": score}, val_iteration)

        # Save checkpoint for best performing model
        if score > best_score:
            log[args.log_name].info("Saving best score with {:.3f}".format(score))
            best_score = score
            path = "./log/tb_" + args.log_name + "/best_model.pth"
            checkpoint = {}
            checkpoint["actor"] = meta_agent.actor.state_dict()
            checkpoint["dynamic_lr"] = meta_agent.dynamic_lr.data
            torch.save(checkpoint, path)

        # Terminate val
        process_done = 0
        for pid in range(args.n_process):
            if str(pid) + "/train_iteration" in process_dict.keys():
                if process_dict[str(pid) + "/train_iteration"] >= args.max_train_iteration:
                    process_done += 1

        if process_done >= args.n_process:
            import sys
            sys.exit()
