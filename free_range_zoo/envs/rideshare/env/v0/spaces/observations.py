"""Observation space constructors for rideshare simulation environment."""
from typing import Tuple, List
import functools
import torch
import free_range_rust
from free_range_rust import Space


def build_observation_space(
    environment_task_counts: torch.IntTensor,
    num_agents: int,
    agent_high: Tuple[int],
    passenger_high: Tuple[int],
) -> List[free_range_rust.Space]:
    """
    Build the observation space for all environments in a batched environment.

    Args:
        environment_task_counts: torch.Tensor - The number of tasks in each environment
        num_agents: int - The number of agents in the environment
        agent_high: Tuple[int] - The high values for the agent observation space
        passenger_high: Tuple[int] - The high values for the fire observation space
    Returns:
        List[free_range_rust.Space] - The observation spaces for the environments
    """
    environment_task_counts = environment_task_counts.tolist()
    return [
        build_single_observation_space(agent_high, passenger_high, task_count, num_agents)
        for task_count in environment_task_counts
    ]


@functools.lru_cache(maxsize=100)
def build_single_observation_space(
    agent_high: Tuple[int],
    passenger_high: Tuple[int],
    num_tasks: int,
    num_agents: int,
) -> free_range_rust.Space:
    """
    Build the observation space for a single environment.

    Args:
        agent_high: Tuple[int] - The high values for the agent observation space
        passenger_high: Tuple[int] - The high values for the fire observation space
        num_tasks: int - The number of tasks in the environment
        num_agents: int - The number of agents in the environment
    Returns:
        gymnasium.Space - The observation space for the environment
    """
    return Space.Dict({
        'self': build_single_agent_observation_space(agent_high),
        'others': Space.Tuple([*[build_single_agent_observation_space(agent_high) for _ in range(num_agents - 1)]]),
        'tasks': build_single_passenger_observation_space(passenger_high, num_tasks),
    })


@functools.lru_cache(maxsize=100)
def build_single_agent_observation_space(high: Tuple[int]):
    """
    Build the observation space for a single agent.

    The agent observation space is defined as follows:
        - (y, x, num_accepted, num_riding)

    Args:
        high: Tuple[int] - The high values for the agent observation space (y, x, num_accepted, num_riding) if unfiltered
    Returns:
        gymnasium.Space - The observation space for the agent
    """
    return Space.Box(low=[0] * len(high), high=high)


@functools.lru_cache(maxsize=100)
def build_single_passenger_observation_space(high: Tuple[int], num_tasks: int):
    """
    Build the observation space for the fire.

    The passenger observation space is defined as follows:
        - If the task observation space includes the (y, x, y_dest, x_dest, accepted_by, riding_by, fare, entered_step)

    Args:
        high: Tuple[int] - The high values for the passenger observation space (follows defined structure above)
        num_tasks: int - The number of tasks in the environment
    Returns:
        gymnasium.Space - The observation space for the fire
    """
    return Space.Tuple([Space.Box([0] * len(high), high=high) for _ in range(num_tasks)])
