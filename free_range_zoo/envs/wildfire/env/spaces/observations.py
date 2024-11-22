"""Observation space constructors for wildfire simulation environment."""
from typing import Tuple, List
import functools

import numpy as np

import free_range_rust
from free_range_rust import Space


def build_observation_space(environment_task_counts, num_agents: int, agent_high: Tuple[int], fire_high: Tuple[int],
                            include_suppressant: bool, include_power: bool) -> List[free_range_rust.Space]:
    """
    Build the observation space for all environments in a batched environment.

    Args:
        environment_task_counts: torch.Tensor - The number of tasks in each environment
        num_agents: int - The number of agents in the environment
        agent_high: Tuple[int] - The high values for the agent observation space
        fire_high: Tuple[int] - The high values for the fire observation space
        include_suppressant: bool - Whether to include the suppressant in the observation space
        include_power: bool - Whether to include the power in the observation space
    Returns:
        List[free_range_rust.Space] - The observation spaces for the environments
    """
    return [
        build_single_observation_space(agent_high, fire_high, task_count, num_agents, include_suppressant, include_power)
        for task_count in environment_task_counts
    ]


@functools.lru_cache(maxsize=100)
def build_single_observation_space(agent_high: Tuple[int],
                                   fire_high: Tuple[int],
                                   num_tasks: int,
                                   num_agents: int,
                                   include_power: bool = True,
                                   include_suppressant: bool = True) -> free_range_rust.Space:
    """
    Build the observation space for a single environment.

    Args:
        agent_high: Tuple[int] - The high values for the agent observation space (y, x, power, suppressant)
        fire_high: Tuple[int] - The high values for the fire observation space (y, x, level, intensity)
        num_tasks: int - The number of tasks in the environment
        num_agents: int - The number of agents in the environment
        include_power: bool - Whether to include the power in the observation space
        include_suppressant: bool - Whether to include the suppressant in the observation space
    Returns:
        gymnasium.Space - The observation space for the environment
    """
    if include_suppressant and not include_power:
        other_high = agent_high[0], agent_high[1], agent_high[3]
    elif not include_suppressant and include_power:
        other_high = agent_high[0], agent_high[1], agent_high[2]
    elif not include_suppressant and not include_power:
        other_high = agent_high[0], agent_high[1]
    else:
        other_high = agent_high

    return Space.Dict({
        'self': build_single_agent_observation_space(agent_high),
        'others': Space.Tuple([*[build_single_agent_observation_space(other_high) for _ in range(num_agents - 1)]]),
        'tasks': build_single_fire_observation_space(fire_high, num_tasks),
    })


@functools.lru_cache(maxsize=100)
def build_single_agent_observation_space(high: Tuple[int]):
    """
    Build the observation space for a single agent.

    The agent observation space is defined as follows:
        - If the agent observation space includes both the suppressant and the power, the space is (y, x, power, suppressant)
        - If the agent observation space includes the suppressant but not the power, the space is (y, x, suppressant)
        - If the agent observation space includes the power but not the suppressant, the space is (y, x, power)
        - If the agent observation space includes neither the suppressant nor the power, the space is (y, x)

    Args:
        high: Tuple[int] - The high values for the agent observation space (y, x, power, suppressant) if unfiltered
    Returns:
        gymnasium.Space - The observation space for the agent
    """
    return Space.Box(low=[0] * len(high), high=high)


@functools.lru_cache(maxsize=100)
def build_single_fire_observation_space(high: Tuple[int], num_tasks: int):
    """
    Build the observation space for the fire.

    The fire observation space is defined as follows:
        - If the fire observation space includes the level and intensity, the space is (y, x, level, intensity)

    Args:
        high: Tuple[int] - The high values for the fire observation space (y, x, level, intensity) if unfiltered
        num_tasks: int - The number of tasks in the environment
    Returns:
        gymnasium.Space - The observation space for the fire
    """
    return Space.Tuple([Space.Box([0] * len(high), high=high) for _ in range(num_tasks)])
