"""Observation space constructors for wildfire simulation environment."""
from typing import Tuple, List
import functools

import free_range_rust
from free_range_rust import Space


def build_observation_space(environment_task_counts, num_agents: int, agent_high: Tuple[int], fire_high: Tuple[int],
                            include_mask: Tuple[bool, bool, bool, bool, bool]) -> List[free_range_rust.Space]:
    """
    Build the observation space for all environments in a batched environment.

    Args:
        environment_task_counts (torch.Tensor): The number of tasks in each environment
        num_agents (int): The number of agents in the environment
        agent_high (Tuple[int]): The high values for the agent observation space
        fire_high (Tuple[int]): The high values for the fire observation space
        include_mask (Tuple[bool, bool, bool, bool, bool]): Mask for including (power, suppressant, range, capacity, equipment) in the observation space.

    Returns:
        List[free_range_rust.Space]: The observation spaces for the environments
    """
    return [
        build_single_observation_space(agent_high, fire_high, task_count, num_agents, include_mask)
        for task_count in environment_task_counts
    ]


@functools.lru_cache(maxsize=100)
def build_single_observation_space(
    agent_high: Tuple[int],
    fire_high: Tuple[int],
    num_tasks: int,
    num_agents: int,
    include_mask: Tuple[bool, bool, bool, bool, bool] = (True, True, False, False, False)) -> free_range_rust.Space:
    """
    Build the observation space for a single environment.

    Args:
        agent_high (Tuple[int]): The high values for the agent observation space (y, x, power, suppressant)
        fire_high (Tuple[int]): The high values for the fire observation space (y, x, level, intensity)
        num_tasks (int): The number of tasks in the environment
        num_agents (int): The number of agents in the environment
        include_mask (Tuple[bool, bool, bool, bool, bool]): Mask for including (power, suppressant, range, capacity, equipment) in the observation space.

    Returns:
        free_range_rust.Space: The observation space for the environment
    """
    # Full order: [y, x, power, suppressant, range, capacity, equipment]
    mask = [True, True] + list(include_mask)
    other_high = tuple([h for h, m in zip(agent_high, mask) if m])

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
        - The full space is (y, x, power, suppressant, range, capacity, equipment)
        - For 'self', all fields are always included.
        - For 'others', the fields included are determined by the observe_other_* flags (power, suppressant, range, capacity, equipment).
        - The order is always: (y, x, power, suppressant, range, capacity, equipment), with fields masked as needed.

    Args:
        high (Tuple[int]): The high values for the agent observation space (y, x, power, suppressant, range, capacity, equipment) if unfiltered/masked

    Returns:
        free_range_rust.Space: The observation space for the agent, with fields masked according to the observation flags for 'others', and all fields for 'self'.
    """
    return Space.Box(low=[0] * len(high), high=high)


@functools.lru_cache(maxsize=100)
def build_single_fire_observation_space(high: Tuple[int], num_tasks: int):
    """
    Build the observation space for the fire.

    The fire observation space is defined as follows:
        - If the fire observation space includes the level and intensity, the space is (y, x, level, intensity)

    Args:
        high (Tuple[int]): The high values for the fire observation space (y, x, level, intensity) if unfiltered
        num_tasks (int): The number of tasks in the environment

    Returns:
        free_range_rust.Space: The observation space for the fire, as a Space.Tuple of Box spaces (one per fire task)
    """
    return Space.Tuple([Space.Box([0] * len(high), high=high) for _ in range(num_tasks)])
