import gym
from gym.envs.registration import register


register(
    id='IPD-v0',
    entry_point='gym_env.ipd.ipd_env:IPDEnv',
    kwargs={'args': None},
    max_episode_steps=150
)

register(
    id='RPS-v0',
    entry_point='gym_env.rps.rps_env:RPSEnv',
    kwargs={'args': None},
    max_episode_steps=150
)


def make_env(args):
    env = gym.make(args.env_name, args=args)
    env._max_episode_steps = args.ep_horizon
    return env
