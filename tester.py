import torch
from meta.peer import Peer
from misc.rl_utils import collect_trajectory
from misc.utils import log_performance
from gym_env import make_env


def meta_test(meta_agent, log, tb_writer, args):
    # Initialize test_iteration
    test_iteration = 0

    # Set env
    env = make_env(args)
    env.seed(args.seed)

    # Load best meta-agent
    path = "./log/tb_" + args.log_name[:-8] + "_log/best_model.pth"
    checkpoint = torch.load(path)
    meta_agent.actor.load_state_dict(checkpoint["actor"])
    meta_agent.dynamic_lr.data = checkpoint["dynamic_lr"]

    # Initialize thread-specific peer
    peer = Peer(log, tb_writer, args, name="peer", i_agent=1)

    # Set agents
    agents = [meta_agent, peer] 

    # Get meta-test persona
    personas = env.sample_personas(is_train=False, is_val=False)

    for i_persona, persona in enumerate(personas): 
        peer.set_persona(persona)
        actors = [agent.actor for agent in agents]

        for i_joint in range(args.chain_horizon + 1):
            # Collect trajectory
            memory, scores = collect_trajectory(agents, actors, env, args)
            log_performance(scores, log, tb_writer, args, i_joint, test_iteration, 0, is_train=False)

            # Perform inner-loop update
            phis = []
            for agent, actor in zip(agents, actors):
                phi = agent.inner_update(actor, memory, i_joint, is_train=False)
                phis.append(phi)

            # For next iteration
            actors = phis

    test_iteration += 1
