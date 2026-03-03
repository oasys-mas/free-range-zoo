"""Create action spaces for the wildfire environment."""
from typing import List
import functools

import torch
import free_range_rust
from free_range_rust import Space


def build_action_space(environment_task_counts: torch.Tensor) -> List[free_range_rust.Space]:
    """
    Build the action space for all environments in a batched environment.

    Args:
        environment_task_counts: torch.Tensor - The number of tasks in each environment
    Returns:
        List[free_range_rust.Space] - The action spaces for the environments
    """
    environment_task_counts = environment_task_counts.tolist()
    return Space.Vector([build_single_action_space(task_count) for task_count in environment_task_counts])


@functools.lru_cache(maxsize=100)
def build_single_action_space(num_tasks_in_environment: int) -> free_range_rust.Space:
    """
    Build the action space for a single environment.

    Action Space structure is defined as follows:
        - If there are no tasks in the environment, the action space is a single action with a value of -1 (noop)
        - If there are tasks in the environment, the action space is a single action for each task, with an additional
          action for noop

    Args:
        num_tasks_in_environment: int - The number of tasks in the environment
    Returns:
        free_range_rust.Space - The action space for the environment
    """
    if num_tasks_in_environment == 0:
        return Space.OneOf([Space.Discrete(1, start=-1)])

    return Space.OneOf([*[Space.Discrete(1, start=0) for _ in range(num_tasks_in_environment)], Space.Discrete(1, start=-1)])
