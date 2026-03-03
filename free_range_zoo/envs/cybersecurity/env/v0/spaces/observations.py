"""Functions for building observation spaces for the cybersecurity environment."""
from typing import Tuple, List, Literal, Union
import functools

import numpy as np

import free_range_rust
from free_range_rust import Space


@functools.lru_cache(maxsize=100)
def build_observation_space(
    agent_type: Union[Literal['attacker'], Literal['defender']],
    num_nodes: int,
    parallel_envs: int,
    num_attackers: int,
    num_defenders: int,
    attacker_high: Tuple[int],
    defender_high: Tuple[int],
    network_high: Tuple[int],
    include_power: bool,
    include_presence: bool,
    include_location: bool,
) -> List[free_range_rust.Space]:
    """
    Build the observation space for all environments in a batched environment.

    Args:
        agent_type: Literal['attacker', 'defender'] - The type of agent for which to build the observation
        num_nodes: int - The number of nodes in the environment
        parallel_envs: int - The number of parallel environments
        num_attackers: int - The number of attackers in the environment
        num_defenders: int - The number of defenders in the environment
        attacker_high: Tuple[int] - The high values for the attacker observation space (power, presence, location)
        defender_high: Tuple[int] - The high values for the defender observation space (mitigation, presence, location)
        network_high: Tuple[int] - The high values for the network observation space (state)
        include_power: bool - Whether to include the power in the observation space
        include_presence: bool - Whether to include the presence in the observation space
        include_location: bool - Whether to include the location in the observation space
    Returns:
        List[gymnasium.Space] - The observation spaces for the environments
    """
    match agent_type:
        case 'defender':
            space = build_single_defender_observation_space(
                defender_high=defender_high,
                network_high=network_high,
                num_tasks=num_nodes,
                num_agents=num_defenders,
                include_power=include_power,
                include_presence=include_presence,
                include_location=include_location,
            )
        case 'attacker':
            space = build_single_attacker_observation_space(
                attacker_high=attacker_high,
                network_high=network_high,
                num_tasks=num_nodes,
                num_agents=num_attackers,
                include_power=include_power,
                include_presence=include_presence,
            )
        case _:
            raise ValueError(f'Invalid agent type: {agent_type}')

    return [space] * parallel_envs


@functools.lru_cache(maxsize=100)
def build_single_attacker_observation_space(attacker_high: Tuple[int],
                                            network_high: Tuple[int],
                                            num_tasks: int,
                                            num_agents: int,
                                            include_power: bool = True,
                                            include_presence: bool = True) -> free_range_rust.Space:
    """
    Build the observation space for a single environment.

    Args:
        attacker_high: Tuple[int] - The high values for the agent observation space (power, presence, location)
        network_high: Tuple[int] - The high values for the network observation space (state)
        num_tasks: int - The number of tasks (subnetworks) in the environment
        num_agents: int - The number of agents in the environment
        include_power: bool - Whether to include the threat in the observation space
        include_presence: bool - Whether to include the presence in the observation space
    Returns:
        free_range_rust.Space - The observation space for the environment
    """
    other_high = ()
    if include_power:
        other_high += attacker_high[0],
    if include_presence:
        other_high += attacker_high[1],

    return Space.Dict({
        'self': build_single_agent_observation_space(attacker_high),
        'others': Space.Tuple([*[build_single_agent_observation_space(other_high) for _ in range(num_agents - 1)]]),
        'tasks': build_single_subnetwork_observation_space(network_high, num_tasks),
    })


@functools.lru_cache(maxsize=100)
def build_single_defender_observation_space(defender_high: Tuple[int],
                                            network_high: Tuple[int],
                                            num_tasks: int,
                                            num_agents: int,
                                            include_power: bool = True,
                                            include_presence: bool = True,
                                            include_location: bool = True) -> free_range_rust.Space:
    """
    Build the observation space for a single environment.

    Args:
        defender_high: Tuple[int] - The high values for the agent observation space (power, presence, location)
        network_high: Tuple[int] - The high values for the network observation space (state)
        num_tasks: int - The number of tasks (subnetworks) in the environment
        num_agents: int - The number of agents in the environment
        include_power: bool - Whether to include the mitigation in the observation space
        include_presence: bool - Whether to include the presence in the observation space
        include_location: bool - Whether to include the location in the observation space
    Returns:
        free_range_rust.Space - The observation space for the environment
    """
    other_high = ()
    if include_power:
        other_high += defender_high[0],
    if include_presence:
        other_high += defender_high[1],
    if include_location:
        other_high += defender_high[2],

    return Space.Dict({
        'self': build_single_agent_observation_space(defender_high),
        'others': Space.Tuple([*[build_single_agent_observation_space(other_high) for _ in range(num_agents - 1)]]),
        'tasks': build_single_subnetwork_observation_space(network_high, num_tasks),
    })


@functools.lru_cache(maxsize=100)
def build_single_agent_observation_space(high: Tuple[int]):
    """
    Build the observation space for a single agent.

    Args:
        high: Tuple[int] - The high values for the agent observation space (power, presence, location) if unfiltered
    Returns:
        free_range_rust.Space - The observation space for the agent
    """
    if len(high) == 0:
        return Space.Discrete(0, start=0)
    return Space.Box(low=[0] * len(high), high=[int(i) for i in high])


@functools.lru_cache(maxsize=None)
def build_single_subnetwork_observation_space(high: Tuple[int], num_tasks: int):
    """
    Build the observation space for the subnetworks in an environment.

    Args:
        high: Tuple[int] - The high values for the subnetwork observation space (state) if unfiltered
        num_tasks: int - The number of tasks in the environment
    Returns:
        free_range_rust.Space - The observation space for the fire
    """
    return Space.Tuple([Space.Box(low=[0] * len(high), high=high) for _ in range(num_tasks)])
