"""Create action spaces for the wildfire environment."""
from typing import List
import functools

import torch
from gymnasium.spaces import Discrete, OneOf
import gymnasium


def build_action_space(environment_task_counts: torch.Tensor) -> List[gymnasium.Space]:
    """
    Build the action space for all environments in a batched environment.

    Args:
        environment_task_counts: torch.Tensor - The number of tasks in each environment
    Returns:
        List[gymnasium.Space] - The action spaces for the environments
    """
    environment_task_counts = environment_task_counts.tolist()
    return [build_single_action_space(task_count) for task_count in environment_task_counts]


@functools.lru_cache(maxsize=100)
def build_single_action_space(num_tasks_in_environment: int) -> gymnasium.Space:
    """
    Build the action space for a single environment.

    Action Space structure is defined as follows:
        - If there are no tasks in the environment, the action space is a single action with a value of -1 (noop)
        - If there are tasks in the environment, the action space is a single action for each task, with an additional
          action for noop

    Args:
        num_tasks_in_environment: int - The number of tasks in the environment
    Returns:
        gymnasium.Space - The action space for the environment
    """
    if num_tasks_in_environment == 0:
        return OneOf([Discrete(1, start=-1)])

    return OneOf([*[Discrete(1) for _ in range(num_tasks_in_environment)], Discrete(1, start=-1)])
