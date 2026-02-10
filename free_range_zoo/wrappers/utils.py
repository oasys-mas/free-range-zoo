"""Wrapper Utility Functions for FreeRangeZoo.

This module provides helper functions for working with wrapped
FreeRangeZoo environments, including utilities to inspect the
wrapper chain applied to an environment.

Functions:
    list_wrappers: Returns an ordered list of all wrappers applied to an environment.

Example Usage:
    >>> from free_range_zoo.wrappers.utils import list_wrappers
    >>> wrappers = list_wrappers(wrapped_env)
    >>> print(wrappers)
    ['OrderEnforcingWrapper', 'ActionSpaceValidatorModifier']
"""
from typing import List
from free_range_zoo.envs._base.v0.env import BatchedAECEnv
from free_range_zoo.wrappers.wrapper_util import shared_wrapper_aec, shared_wrapper_gym, shared_wrapper_parr


def list_wrappers(env: BatchedAECEnv) -> List[str]:
    """Return an ordered list of all wrappers applied to a PettingZoo environment.

    Ordered from outermost wrapper to innermost.

    Args:
        env (BatchedAECEnv): A free-range-zoo wrapped environment.

    Returns:
        List[str]: The list of wrapper class names. For shared_wrapper types, includes the modifier class name.
    """
    wrappers = []
    current = env

    # Walk down the chain of wrappers
    while hasattr(current, "env") or hasattr(current, "aec_env"):
        wrappers.append(type(current).__name__)
        if isinstance(current, shared_wrapper_parr) or \
                isinstance(current, shared_wrapper_gym) or \
                isinstance(current, shared_wrapper_aec):
            wrappers[-1] = wrappers[-1] + f":{current.modifier_class}"

        current = current.env if hasattr(current, "env") else current.aec_env

    # current is now the base environment, not a wrapper
    return wrappers
    wrappers = []
    current = env

    # Walk down the chain of wrappers
    while hasattr(current, "env") or hasattr(current, "aec_env"):
        wrappers.append(type(current).__name__)
        if isinstance(current, shared_wrapper_parr) or \
                isinstance(current, shared_wrapper_gym) or \
                isinstance(current, shared_wrapper_aec):
            wrappers[-1] = wrappers[-1] + f":{current.modifier_class}"

        current = current.env if hasattr(current, "env") else current.aec_env

    # current is now the base environment, not a wrapper
    return wrappers
