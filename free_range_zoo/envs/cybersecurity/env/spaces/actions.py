"""Contains functions for building action spaces for the cybersecurity environment."""

from typing import List, Literal, Union
import functools

import torch
from gymnasium.spaces import Discrete, OneOf
import gymnasium


def build_action_space(agent_type: Union[Literal['attacker'], Literal['defender']], show_bad_actions: bool,
                       environment_task_counts: torch.IntTensor, current_location: torch.IntTensor) -> List[gymnasium.Space]:
    """
    Build the action space for all environments in a batched environment.

    Args:
        agent_type: Literal['attacker', 'defender'] - The type of agent for which to build the action
        show_bad_actions: bool - Whether to include bad actions in the action space
        environment_task_counts: torch.IntTensor - The number of tasks in each environment
        current_location: torch.IntTensor - The current location of the subject agent in each environment
    Returns:
        List[gymnasium.Space] - The action spaces for the environments
    """
    match agent_type:
        case 'defender':
            info = zip(environment_task_counts.tolist(), current_location.tolist())
            space = [build_single_defender_action_space(count, loc, show_bad_actions) for count, loc in info]
        case 'attacker':
            space = [build_single_attacker_action_space(task_count) for task_count in environment_task_counts]
        case _:
            raise ValueError(f'Invalid agent type: {agent_type}')

    return space


@functools.lru_cache(maxsize=100)
def build_single_defender_action_space(num_tasks_in_environment: int, current_location: int,
                                       show_bad_actions: bool) -> gymnasium.Space:
    """
    Build the action space for a single defender agent in the environment.

    Args:
        num_tasks_in_environment: int - The number of tasks in the environment (number of accessible subnetworks + 1)
        current_location: int - The current location of the subject agent, -1 indicates the home node
        show_bad_actions: bool - Whether to include bad actions in the action space
    Returns:
        gymnasium.Space - The action space for the environment
    """
    # The agent is not present in the environment so the only action available is to noop
    if num_tasks_in_environment == 0:
        return OneOf([Discrete(1, start=-1)])  # noop

    # The agent is at the home node so they do not have the option to patch if bad options are not shown
    if show_bad_actions and current_location == -1:
        return OneOf([
            *[Discrete(1) for _ in range(num_tasks_in_environment)],  # move to connected nodes
            Discrete(1, start=-1),  # noop
            Discrete(1, start=-2),  # patch current node
            Discrete(1, start=-3),  # monitor
        ])
    elif current_location == -1:
        return OneOf([
            *[Discrete(1) for _ in range(num_tasks_in_environment)],  # move to connected nodes
            Discrete(1, start=-1),  # noop
            Discrete(1, start=-3),  # monitor
        ])

    return OneOf([
        *[Discrete(1) for _ in range(num_tasks_in_environment - 1)],  # move to connected nodes
        Discrete(1, start=-1),  # noop
        Discrete(1, start=-2),  # patch current node
        Discrete(1, start=-3),  # monitor
    ])


@functools.lru_cache(maxsize=100)
def build_single_attacker_action_space(num_tasks_in_environment: int) -> gymnasium.Space:
    """
    Build the action space for a single attacker agent in the environment.

    Args:
        num_tasks_in_environment: int - The number of tasks in the environment (number of subnetworks)
    Returns:
        gymnasium.Space - The action space for the environment
    """
    # The agent is not present in the environment so the only action available is to noop
    if num_tasks_in_environment == 0:
        return OneOf([Discrete(1, start=-1)])

    return OneOf([
        *[Discrete(1) for _ in range(num_tasks_in_environment)],  # attack node
        Discrete(1, start=-1),  # noop
    ])
