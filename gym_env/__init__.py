import gym
from gym.envs.registration import register
from gym_env.multi_mujoco.multiagent_mujoco import MultiAgentMuJoCo
from gym_env.multiprocessing_env import SubprocVecEnv


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


register(
    id='HalfCheetahDir-v0',
    entry_point='gym_env.multi_mujoco.half_cheetah_dir_env:HalfCheetahDirEnv',
    kwargs={'args': None},
    max_episode_steps=200
)


def make_env(args):
    if args.env_name == "HalfCheetahDir-v0":
        def _make_env():
            env = MultiAgentMuJoCo(args=args)
            return env
        env = SubprocVecEnv([_make_env for _ in range(args.traj_batch_size)])
    else:
        env = gym.make(args.env_name, args=args)
        env._max_episode_steps = args.ep_horizon
    return env
