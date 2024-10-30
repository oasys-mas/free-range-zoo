"""Profile the cybersecurity environment."""

import sys

sys.path.append('.')

import argparse
import torch
import cProfile
from pstats import Stats

from free_range_zoo.envs import cybersecurity_v0, wildfire_v0, rideshare_v0
from tests.utils import cybersecurity_configs, wildfire_configs, rideshare_configs


def main():
    """Profile the cybersecurity environment."""
    args = parse_args()
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    match args.environment:
        case 'wildfire':
            configuration = wildfire_configs.non_stochastic()
            env = wildfire_v0.parallel_env(
                parallel_envs=args.parallel_envs,
                max_steps=args.steps,
                configuration=configuration,
                device=device,
                buffer_size=args.buffer_size,
            )
        case 'cybersecurity':
            configuration = cybersecurity_configs.non_stochastic()
            env = cybersecurity_v0.parallel_env(
                parallel_envs=args.parallel_envs,
                max_steps=args.steps,
                configuration=configuration,
                device=device,
                buffer_size=args.buffer_size,
            )
        case 'rideshare':
            configuration = rideshare_configs.non_stochastic()
            env = rideshare_v0.parallel_env(
                parallel_envs=args.parallel_envs,
                max_steps=args.steps,
                configuration=configuration,
                device=device,
            )
        case _:
            raise ValueError(f'Invalid environment: {args.environment}')

    env.reset()

    profiler = cProfile.Profile()

    current_step = 1
    while not torch.all(env.finished):
        action = {}

        for agent in env.agents:
            profiler.enable()
            env.observation_space(agent)
            action_spaces = env.action_space(agent)
            profiler.disable()

            actions = []
            for action_space in action_spaces:
                actions.append(action_space.sample())
            actions = torch.tensor(actions, dtype=torch.int32, device=device)
            action[agent] = actions

        profiler.enable()
        observation, reward, term, trunc, info = env.step(action)
        profiler.disable()

        print(f'Completed step {current_step}')
        current_step += 1

    profiler.create_stats()

    stats = Stats(profiler).strip_dirs().sort_stats(*args.sort_by)
    stats.print_stats(args.amount)

    if args.output is not None:
        stats.dump_stats(args.output)


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser('Profile a free-range-zoo environment')

    parser.add_argument('environment',
                        type=str,
                        choices=['wildfire', 'cybersecurity', 'rideshare'],
                        help='The environment to profile')
    parser.add_argument('--parallel_envs', type=int, default=10, help='The number of parallel environments')
    parser.add_argument('--steps', type=int, default=15, help='The number of steps to run')
    parser.add_argument('--buffer_size', type=int, default=0, help='The size of the random buffer to use')

    parser.add_argument('--sort_by',
                        type=str,
                        nargs='*',
                        choices=['calls', 'cumtime', 'file', 'ncalls', 'pcalls', 'line', 'name', 'nfl', 'stdname', 'tottime'],
                        default=['tottime'],
                        help='The way to sort the results')
    parser.add_argument('--amount', type=float, default=10, help='The amount of data to display')
    parser.add_argument('--string_filter', type=str, default=None, help='Filter the results by a string')
    parser.add_argument('--output', type=str, default=None, help='The output file to write the stats to')

    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')

    return parser.parse_args()


if __name__ == '__main__':
    main()
