from typing import List
import functools
import cachetools

import torch
from gymnasium.spaces import Discrete, OneOf
import gymnasium

from free_range_zoo.utils.caching import optimized_convert_hashable


@cachetools.cached(cache=cachetools.LRUCache(float('inf')),
                   key=lambda environment_task_counts, *args, **kwargs: optimized_convert_hashable(environment_task_counts),
                   info=True)
def build_action_space(environment_task_counts: torch.Tensor) -> List[gymnasium.Space]:
    """
    Builds the action space for all environments in a batched environment

    Args:
        environment_task_counts: torch.Tensor - The number of tasks in each environment
    Returns:
        List[gymnasium.Space] - The action spaces for the environments
    """
    return [build_single_action_space(task_count) for task_count in environment_task_counts]


@functools.lru_cache(maxsize=None)
def build_single_action_space(num_tasks_in_environment: int) -> gymnasium.Space:
    """
    Builds the action space for a single environment

    Args:
        num_tasks_in_environment: int - The number of tasks in the environment
    Returns:
        gymnasium.Space - The action space for the environment
    """
    if num_tasks_in_environment == 0:
        return OneOf([Discrete(1, start=-1)])

    return OneOf([*[Discrete(1) for _ in range(num_tasks_in_environment)], Discrete(1, start=-1)])
