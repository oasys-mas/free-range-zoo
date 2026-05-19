"""Action Space Validator for Environment Actions.

This module provides the ActionSpaceValidatorModifier class for validating
that actions are within the bounds of each action space.
"""
from supersuit.generic_wrappers.utils.base_modifier import BaseModifier

import torch
import logging

from free_range_zoo.envs._base.v0.env import BatchedAECEnv
from free_range_zoo.wrappers.wrapper_util import shared_wrapper

logger = logging.getLogger('free_range_zoo')


class ActionSpaceValidatorModifier(BaseModifier):
    """Wrapper for validating actions are within the bounds of each action space."""

    env = True
    subject_agent = True

    def __init__(
        self,
        env: BatchedAECEnv,
        subject_agent: str,
        allow_flexible_task_tags: bool = True,
        cast_invalid_to_noop: bool = True,
    ):
        """Initialize the ActionSpaceValidatorModifier.

        Args:
            env (BatchedAECEnv): The environment to wrap.
            subject_agent (str): The subject agent of the graph wrapper.
            allow_flexible_task_tags (bool): Whether to allow task-agnostic actions
                to be used for any task instead of just the task-agnostic task.
            cast_invalid_to_noop (bool): Cast invalid actions to noop instead of throwing
        Returns:
            None: None
        """
        self.env = env
        self.allow_flexible_task_tags = allow_flexible_task_tags
        self.cast_invalid_to_noop = cast_invalid_to_noop

        # Unpack the the parallel environment if it is wrapped in one.
        if hasattr(self.env, 'aec_env'):
            self.env = self.env.aec_env
        # Unpack the order enforcing wrapper if it has one of those.
        if hasattr(self.env, 'env'):
            self.env = self.env.env

        self.subject_agent = subject_agent

    def modify_action(self, actions: torch.IntTensor) -> torch.IntTensor:
        """Modify the action before it is passed to the environment.

        Args:
            actions (torch.IntTensor): The action to modify.

        Returns:
            torch.IntTensor: The modified action.
        """
        action_spaces = self.env.action_space(self.subject_agent).spaces
        actions_list = actions.split(1, dim=0)

        for index, (action, space) in enumerate(zip(actions_list, action_spaces)):
            task_channel, action_channel = action.squeeze(0)
            if not self._is_valid_action(space, task_channel, action_channel):
                if self.cast_invalid_to_noop:
                    actions_list[index] = torch.tensor([[0, -1]], dtype=torch.int32)
                else:
                    self._raise_invalid_action(index, action, space)

        return torch.cat(actions_list, dim=0)

    def _is_valid_action(self, space, task_channel: int, action_channel: int) -> bool:
        """Check if an action is within the bounds of the action space."""
        if action_channel < 0 and self.allow_flexible_task_tags:
            return any(action_channel == subspace.start for subspace in reversed(space.spaces))

        try:
            discrete = space.spaces[task_channel]
        except IndexError:
            return False

        return discrete.start <= action_channel <= discrete.start + discrete.n

    def _raise_invalid_action(self, index: int, action: torch.IntTensor, space) -> None:
        logger.critical(f"{self.subject_agent} in batch {index} attempted to take an action on a undefined task."
                        f"\rAction: {action}"
                        f"\rSpace: {space}")
        raise IndexError


def space_validator_wrapper_v0(
    env: BatchedAECEnv,
    allow_flexible_task_tags: bool = True,
    cast_invalid_to_noop: bool = False,
) -> BatchedAECEnv:
    """Apply the ActionSpaceValidatorModifier to the environment.

    Args:
        env (BatchedAECEnv): The environment to wrap.
        allow_flexible_task_tags (bool): Whether to allow task-agnostic actions
            to be used for any task instead of just the task-agnostic task.
        cast_invalid_to_noop (bool): Cast invalid actions to noop instead of throwing

    Returns:
        BatchedAECEnv: The wrapped environment.
    """
    return shared_wrapper(
        env,
        ActionSpaceValidatorModifier,
        allow_flexible_task_tags=allow_flexible_task_tags,
        cast_invalid_to_noop=cast_invalid_to_noop,
    )
