"""Contains functions for building action spaces for the cybersecurity environment."""

from typing import List, Literal, Union
import functools

import torch
import free_range_rust
from free_range_rust import Space


def build_action_space(agent_type: Union[Literal['attacker'],
                                         Literal['defender']], show_bad_actions: bool, environment_task_counts: torch.IntTensor,
                       current_location: torch.IntTensor) -> List[free_range_rust.Space]:
    """
    Build the action space for all environments in a batched environment.

    Args:
        agent_type: Literal['attacker', 'defender'] - The type of agent for which to build the action
        show_bad_actions: bool - Whether to include bad actions in the action space
        environment_task_counts: torch.IntTensor - The number of tasks in each environment
        current_location: torch.IntTensor - The current location of the subject agent in each environment
    Returns:
        List[free_range_rust.Space] - The action spaces for the environments
    """
    environment_task_counts = environment_task_counts.tolist()

    match agent_type:
        case 'defender':
            info = zip(environment_task_counts, current_location.tolist())
            space = Space.Vector([build_single_defender_action_space(count, loc, show_bad_actions) for count, loc in info])
        case 'attacker':
            space = Space.Vector([build_single_attacker_action_space(task_count) for task_count in environment_task_counts])
        case _:
            raise ValueError(f'Invalid agent type: {agent_type}')

    return space


@functools.lru_cache(maxsize=100)
def build_single_defender_action_space(num_tasks_in_environment: int, current_location: int,
                                       show_bad_actions: bool) -> free_range_rust.Space:
    """
    Build the action space for a single defender agent in the environment.

    Args:
        num_tasks_in_environment: int - The number of tasks in the environment (number of accessible subnetworks + 1)
        current_location: int - The current location of the subject agent, -1 indicates the home node
        show_bad_actions: bool - Whether to include bad actions in the action space
    Returns:
        free_range_rust.Space - The action space for the environment
    """
    # The agent is not present in the environment so the only action available is to noop
    if num_tasks_in_environment == 0:
        return Space.OneOf([Space.Discrete(1, start=-1)])  # noop

    # The agent is at the home node so they do not have the option to patch if bad options are not shown
    if show_bad_actions and current_location == -1:
        return Space.OneOf([
            *[Space.Discrete(1, start=0) for _ in range(num_tasks_in_environment)],  # move to connected nodes
            Space.Discrete(1, start=-1),  # noop
            Space.Discrete(1, start=-2),  # patch current node
            Space.Discrete(1, start=-3),  # monitor
        ])
    elif current_location == -1:
        return Space.OneOf([
            *[Space.Discrete(1, start=0) for _ in range(num_tasks_in_environment)],  # move to connected nodes
            Space.Discrete(1, start=-1),  # noop
            Space.Discrete(1, start=-3),  # monitor
        ])

    return Space.OneOf([
        *[Space.Discrete(1, start=0) for _ in range(current_location)],  # move to connected nodes
        Space.Discrete(0, start=0),  # move to current node (empty because this action isn't available)
        *[Space.Discrete(1, start=0) for _ in range(num_tasks_in_environment - current_location - 1)],  # move to connected nodes
        Space.Discrete(1, start=-1),  # noop
        Space.Discrete(1, start=-2),  # patch current node
        Space.Discrete(1, start=-3),  # monitor
    ])


@functools.lru_cache(maxsize=100)
def build_single_attacker_action_space(num_tasks_in_environment: int) -> free_range_rust.Space:
    """
    Build the action space for a single attacker agent in the environment.

    Args:
        num_tasks_in_environment: int - The number of tasks in the environment (number of subnetworks)
    Returns:
        free_range_rust.Space - The action space for the environment
    """
    # The agent is not present in the environment so the only action available is to noop
    if num_tasks_in_environment == 0:
        return Space.OneOf([Space.Discrete(1, start=-1)])

    return Space.OneOf([
        *[Space.Discrete(1, start=0) for _ in range(num_tasks_in_environment)],  # attack node
        Space.Discrete(1, start=-1),  # noop
    ])
