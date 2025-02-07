"""Run one of the cybersecurity based on Random Baseline and generate logs for rendering."""

import sys

sys.path.append('.')

import argparse
import torch

from free_range_zoo.envs import cybersecurity_v0
from tests.utils import cybersecurity_configs
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
from free_range_zoo.envs.cybersecurity.baselines import (
    NoopBaseline,
    RandomBaseline,
    PatchedAttackerBaseline,
    ExploitedAttackerBaseline,
    PatchedDefenderBaseline,
    ExploitedDefenderBaseline,
    CampDefenderBaseline,
)


def main():
    """Run baselines for the cybersecurity environment."""

    device = 'cpu'
    parallel_envs = 2
    attacker_baseline = 'random'
    defender_baseline = 'random'
    configuration = cybersecurity_configs.non_stochastic()
    env = cybersecurity_v0.parallel_env(
        parallel_envs=2,
        max_steps=10,
        configuration=configuration,
        device=device,
        buffer_size=0,
        log_directory="outputs/cyberSec_logging_test_0",
    )
    env = action_mapping_wrapper_v0(env)
    observation, _ = env.reset()

    agents = {}
    for agent_name in env.agents:
        agent_type = agent_name.split('_')[0]
        match agent_type:
            case 'attacker':
                match attacker_baseline:
                    case 'noop':
                        agents[agent_name] = NoopBaseline(agent_name, parallel_envs)
                    case 'random':
                        agents[agent_name] = RandomBaseline(agent_name, parallel_envs)
                    case 'exploited':
                        agent_args = (configuration.network_config.num_states, agent_name, parallel_envs)
                        agents[agent_name] = ExploitedAttackerBaseline(*agent_args)
                    case 'patched':
                        agents[agent_name] = PatchedAttackerBaseline(agent_name, parallel_envs)
            case 'defender':
                match defender_baseline:
                    case 'noop':
                        agents[agent_name] = NoopBaseline(agent_name, parallel_envs)
                    case 'random':
                        agents[agent_name] = RandomBaseline(agent_name, parallel_envs)
                    case 'exploited':
                        agent_args = (configuration.network_config.num_states, agent_name, parallel_envs)
                        agents[agent_name] = ExploitedDefenderBaseline(*agent_args)
                    case 'patched':
                        agents[agent_name] = PatchedDefenderBaseline(agent_name, parallel_envs)
                    case 'camp':
                        agents[agent_name] = CampDefenderBaseline(agent_name, parallel_envs)

    current_step = 1
    while not torch.all(env.finished):
        action = {}

        for agent in env.agents:
            agent_model = agents[agent]
            agent_model.observe(observation[agent])

            actions = agent_model.act(env.action_space(agent))

            actions = torch.tensor(actions, device=device, dtype=torch.int32)
            action[agent] = actions

        observation, reward, term, trunc, info = env.step(action)

        print(f'Completed step {current_step}')
        current_step += 1


if __name__ == '__main__':
    main()
