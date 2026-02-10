"""Wrapper Utilities for FreeRangeZoo Environments.

This module provides wrapper classes and utilities for modifying and
extending FreeRangeZoo environments. Wrappers can transform observations,
actions, and other aspects of environment behavior.

Available Wrappers:
    Action and Task Mapping:
        - action_mapping_wrapper_v0: Maps actions to individual tasks

    Validation:
        - space_validator_wrapper_v0: Validates actions are within action spaces

    Utilities:
        - list_wrappers: Lists all wrappers applied to an environment

Wrapper Utilities:
    - shared_wrapper_aec: AEC environment wrapper application
    - shared_wrapper_parr: Parallel environment wrapper application
    - shared_wrapper_gym: Gymnasium environment wrapper application
    - shared_wrapper: Unified wrapper chooser
    - PassthroughWrapperModifier: Fallback modifier for selective application

Example Usage:
    >>> from free_range_zoo.wrappers import action_mapping_wrapper_v0
    >>> wrapped_env = action_mapping_wrapper_v0(env, subject_agent="firefighter_0")
"""

from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
from free_range_zoo.wrappers.space_validator import space_validator_wrapper_v0
from free_range_zoo.wrappers.utils import list_wrappers
from free_range_zoo.wrappers.wrapper_util import (
    shared_wrapper,
    shared_wrapper_aec,
    shared_wrapper_parr,
    shared_wrapper_gym,
    PassthroughWrapperModifier,
)

__all__ = [
    "action_mapping_wrapper_v0",
    "space_validator_wrapper_v0",
    "list_wrappers",
    "shared_wrapper",
    "shared_wrapper_aec",
    "shared_wrapper_parr",
    "shared_wrapper_gym",
    "PassthroughWrapperModifier",
]