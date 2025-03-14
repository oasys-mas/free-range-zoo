# Evaluation

Below is a example evaluation script for the wildfire domain which will be used to evaluate submitted policies. You will submit a variant of this script for us to evaluate your policy. Please comment your code so it is easy for us to see where you are initializing and loading your model/ weight files.

The evaluation script can be used by running.
```sh
python <path to script> <output_dir> <configuration_path>
```

## Evaluation Script

```python
"""Evaluate wildfire baselines on testing configurations."""

import warnings

warnings.simplefilter('ignore', UserWarning)

import argparse
import os
import sys
import torch
import logging
import pickle

from free_range_zoo.envs import wildfire_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
from free_range_zoo.envs.wildfire.baselines import NoopBaseline, RandomBaseline, StrongestBaseline, WeakestBaseline

FORMAT_STRING = "[%(asctime)s] [%(levelname)8s] [%(name)10s] [%(filename)21s:%(lineno)03d] %(message)s"


def main() -> None:
    """Run the training experiment."""
    global device, args, dataset
    args = handle_args()
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    if args.threads > 1:
        torch.set_num_threads(args.threads)

    os.makedirs(args.output, exist_ok=True)

    if os.path.exists(args.output):
        main_logger.warning(f'Output directory {args.output} already exists and may contain artifacts from a previous run.')

    main_logger.info(f'Running the baseline experiment on device {device} and parameters:')
    for key, value in vars(args).items():
        main_logger.info(f'- {key}: {value}')

    torch.use_deterministic_algorithms(True, warn_only=True)
    generator = torch.Generator()
    generator = generator.manual_seed(args.seed)
    torch.manual_seed(torch.randint(0, 100000000, (1, ), generator=generator).item())

    main_logger.info('Initializing dataset')

    try:
        test()
    except KeyboardInterrupt:
        main_logger.warning('Testing interrupted by user')
    except Exception as e:
        main_logger.error(f'Error during testing: {e}')
        raise e


@torch.no_grad()
def test() -> None:
    """
    Run the testing episodes for the model.

    Args:
        model: nn.Module - The model to validate.
    """
    # TODO: For other domains, swap environment initialization and agent names
    agents = ['firefighter_1', 'firefighter_2', 'firefighter_3']
    env = wildfire_v0.parallel_env(
        parallel_envs=args.testing_episodes,
        max_steps=50,
        device=device,
        configuration=pickle.load(args.config),
        buffer_size=50,
        single_seeding=True,
        show_bad_actions=False,
    )

    env = action_mapping_wrapper_v0(env)
    observation, _ = env.reset(seed=0)

    agents = {}
    for agent_name in env.agents:
        agents[agent_name] = # TODO: Call policy class initialization to initialize each agent

    step = 0
    total_rewards = {agent: torch.zeros(args.testing_episodes) for agent in agents}
    while not torch.all(env.finished):
        test_logger.info(f'STEP {step}')
        agent_actions = {}
        for agent_name, agent_model in agents.items():
            agent_model.observe(observation[agent_name])

            actions = agent_model.act(env.action_space(agent_name))
            actions = torch.tensor(actions, device=device, dtype=torch.int32)
            agent_actions[agent_name] = actions

        observation, reward, term, trunc, info = env.step(agent_actions)

        test_logger.info('ACTIONS')
        for batch in range(args.testing_episodes):
            batch_actions = ' '.join(f'{agent_name}: {str(agent_actions[batch].tolist()):<10}\t'
                                     for agent_name, agent_actions in agent_actions.items())
            test_logger.info(f'{batch + 1}:\t{batch_actions}')

        test_logger.info('REWARDS')
        for agent_name in env.agents:
            test_logger.info(f'{agent_name}: {reward[agent_name]}')
            total_rewards[agent_name] += reward[agent_name]

        step += 1

    for agent_name, total_reward in total_rewards.items():
        global_total_rewards[agent_name] += total_reward

    test_logger.info(logging.create_log_heading('TOTAL REWARDS'))
    for agent_name, total_reward in total_rewards.items():
        test_logger.info(f'{agent_name}: {total_reward}')

    totals = torch.zeros(args.testing_episodes)
    for agent, reward_tensor in total_rewards.items():
        totals += reward_tensor

    total_mean = round(totals.mean().item(), 3)
    total_std_dev = round(totals.std().item(), 3)

    average_mean = round(total_mean / len(agents), 3)
    average_std_dev = round(total_std_dev / len(agents), 3)

    test_logger.info(logging.create_log_heading('REWARD SUMMARY'))
    test_logger.info(f'Average Reward: {average_mean} ± {average_std_dev}')
    test_logger.info(f'Total Reward: {total_mean} ± {total_std_dev}')


def handle_args() -> argparse.Namespace:
    """
    Handle script arguments.

    Returns:
        argparse.Namespace - parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Run baseline policies on a given wildfire configuration.')

    general = parser.add_argument_group('General')
    general.add_argument('output', type=str, help='output directory for all experiments artifacts')
    general.add_argument('config', type=str, help='path to environment configuration to utilize')
    general.add_argument('--cuda', action='store_true', help='Utilize cuda if available')
    general.add_argument('--threads', type=int, default=1, help='utilize this many threads for the experiment')

    reproducible = parser.add_argument_group('Reproducibility')
    reproducible.add_argument('--seed', type=int, default=None, help='seed for the experiment')
    reproducible.add_argument('--dataset_seed', type=int, default=None, help='seed for initializing the configuration dataset')

    validation = parser.add_argument_group('Validation')
    validation.add_argument('--testing_episodes', type=int, default=16, help='number of episodes to run per test')

    return parser.parse_args()


main_logger = logging.get('main')
test_logger = logging.get('baseline')

if __name__ == '__main__':
    main()
    sys.exit()
```
