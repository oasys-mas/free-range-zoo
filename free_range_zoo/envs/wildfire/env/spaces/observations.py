from typing import Tuple, List
import cachetools
import functools

import numpy as np

from gymnasium.spaces import Box, Dict, Tuple as TupleSpace
import gymnasium

from free_range_zoo.free_range_zoo.utils.caching import optimized_convert_hashable


@cachetools.cached(cache=cachetools.LRUCache(float('inf')),
                   key=lambda environment_task_counts, *args, **kwargs: optimized_convert_hashable(environment_task_counts),
                   info=True)
def build_observation_space(environment_task_counts,
                            num_agents: int,
                            agent_high: Tuple[int],
                            fire_high: Tuple[int],
                            include_suppressant: bool,
                            include_power: bool) -> List[gymnasium.Space]:
    """
    Builds the observation space for all environments in a batched environment

    Args:
        environment_task_counts: torch.Tensor - The number of tasks in each environment
        num_agents: int - The number of agents in the environment
        agent_high: Tuple[int] - The high values for the agent observation space
        fire_high: Tuple[int] - The high values for the fire observation space
        include_suppressant: bool - Whether to include the suppressant in the observation space
        include_power: bool - Whether to include the power in the observation space
    Returns:
        List[gymnasium.Space] - The observation spaces for the environments
    """
    return [build_single_observation_space(agent_high,
                                           fire_high,
                                           task_count,
                                           num_agents,
                                           include_suppressant,
                                           include_power) for task_count in environment_task_counts]


@functools.lru_cache(maxsize=None)
def build_single_observation_space(agent_high: Tuple[int],
                                   fire_high: Tuple[int],
                                   num_tasks: int,
                                   num_agents: int,
                                   include_power: bool = True,
                                   include_suppressant: bool = True) -> gymnasium.Space:
    """
    Builds the observation space for a single environment

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

    return Dict({
        'self': build_single_agent_observation_space(agent_high),
        'others': TupleSpace([*[build_single_agent_observation_space(other_high) for _ in range(num_agents - 1)]]),
        'fire': build_single_fire_observation_space(fire_high, num_tasks),
    })


@functools.lru_cache(maxsize=None)
def build_single_agent_observation_space(high: Tuple[int]):
    """
    Builds the observation space for a single agent

    Args:
        high: Tuple[int] - The high values for the agent observation space (y, x, power, suppressant) if unfiltered
    Returns:
        gymnasium.Space - The observation space for the agent
    """
    return Box(low=np.array([0] * len(high)), high=np.array(high), dtype=np.float32)


@functools.lru_cache(maxsize=None)
def build_single_fire_observation_space(high: Tuple[int], num_tasks: int):
    """
    Builds the observation space for the fire

    Args:
        high: Tuple[int] - The high values for the fire observation space (y, x, level, intensity) if unfiltered
        num_tasks: int - The number of tasks in the environment
    Returns:
        gymnasium.Space - The observation space for the fire
    """
    return TupleSpace([Box(low=np.array([0] * len(high)), high=np.array(high), dtype=np.float32) for _ in range(num_tasks)])
