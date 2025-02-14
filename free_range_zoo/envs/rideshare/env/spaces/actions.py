"""Create action spaces for the rideshare environment."""
from typing import Tuple, List
import functools

import torch
import free_range_rust
from free_range_rust import Space


def build_action_space(environment_action_choices: torch.IntTensor,
                       environment_task_counts: torch.IntTensor) -> List[free_range_rust.Space]:
    """
    Build the action space for all environments in a batched environment.

    Args:
        environment_action_choices: torch.Tensor - The number of tasks in each environment
        environment_task_counts: torch.Tensor - The number of tasks in each environment
    Returns:
        List[free_range_rust.Space] - The action spaces for the environments
    """
    spaces, offset = [], 0
    for task_count in environment_task_counts:
        tasks = tuple(environment_action_choices[offset:offset + task_count, 1].tolist())
        spaces.append(build_single_action_space(tasks))

    return Space.Vector(spaces)


@functools.lru_cache(maxsize=100)
def build_single_action_space(action_choices: Tuple[int]) -> free_range_rust.Space:
    """
    Build the action space for a single environment.

    Action space structure is defined as follows:
        - If there are no tasks in the environment, the action space is a single action with a value of -1 (noop)
        - If there are tasks in the environment, the action space is a single action for each task, with an additional
          action for noop (-1).
        - (1) describes an accept, (2) describes moving picking up a passenger, (3) represents moving to drop off a
          passenger.

    Args:
        action_choices: Tuple[int] - A class encoding of the action choice for each task within the environment.
    Returns:
        free_range_rust.Space - The action space for the environment
    """
    if len(action_choices) == 0:
        return Space.OneOf([Space.Discrete(1, start=-1)])

    return Space.OneOf([*[Space.Discrete(1, start=choice) for choice in action_choices], Space.Discrete(1, start=-1)])
