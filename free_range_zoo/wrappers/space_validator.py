"""Action space validator to verify actions are within the bounds of each action space."""
from supersuit.generic_wrappers.utils.base_modifier import BaseModifier

import torch
import logging

from free_range_zoo.wrappers.wrapper_util import shared_wrapper
from free_range_zoo.utils.env import BatchedAECEnv

logger = logging.getLogger('free_range_zoo')


class ActionSpaceValidatorModifier(BaseModifier):
    """Wrapper for validating actions are within the bounds of each action space."""

    env = True
    subject_agent = True

    def __init__(self, env: BatchedAECEnv, subject_agent: str, allow_flexible_task_tags: bool = True):
        """
        Initialize the ActionSpaceValidatorModifier.

        Args:
            env: BatchedAECEnv - The environment to wrap.
            subject_agent: str - The subject agent of the graph wrapper.
            allow_flexible_task_tags: bool - Whether to allow task-agnostic actions to be used for any task instead of just the task-agnostic task.
        """
        self.env = env
        self.allow_flexible_task_tags = allow_flexible_task_tags

        # Unpack the the parallel environment if it is wrapped in one.
        if hasattr(self.env, 'aec_env'):
            self.env = self.env.aec_env
        # Unpack the order enforcing wrapper if it has one of those.
        if hasattr(self.env, 'env'):
            self.env = self.env.env

        self.subject_agent = subject_agent

    def modify_action(self, actions: torch.IntTensor):
        """
        Modify the action before it is passed to the environment.

        Args:
            actions: The action to modify.
        """
        actions_iter = actions.split(1, dim=0)
        action_spaces = self.env.action_space(self.subject_agent).spaces

        for index, (action, space) in enumerate(zip(actions_iter, action_spaces)):
            task_channel, action_channel = action.squeeze(0)

            # If the action is a task-agnostic action, we need to check if it is within the bounds of the action space.
            if action_channel < 0 and self.allow_flexible_task_tags:
                is_bad_space = True
                for subspace in reversed(space.spaces):
                    if action_channel == subspace.start:
                        is_bad_space = False
                        break

                if is_bad_space:
                    logger.critical(f'''{self.subject_agent} in batch {index} attempted to take an action on a undefined task.
                        \rAction: {action}
                        \rSpace: {space}''')
                    raise IndexError
            else:
                try:
                    discrete = space.spaces[task_channel]
                except IndexError as e:
                    logger.critical(f'''{self.subject_agent} in batch {index} attempted to take an action on a undefined task.
                        \rAction: {action}
                        \rSpace: {space}''')
                    raise e
                try:
                    if action_channel < discrete.start or action_channel > discrete.start + discrete.n:
                        raise IndexError
                except IndexError as e:
                    logger.critical(f'''{self.subject_agent} in batch {index} attempted to take an action on a undefined task.
                        \rAction: {action}
                        \rSpace: {space}''')
                    raise e

        return actions


def space_validator_wrapper_v0(env: BatchedAECEnv, allow_flexible_task_tags: bool = False) -> BatchedAECEnv:
    """
    Apply the ActionSpaceValidatorModifier to the environment.

    Args:
        env: BatchedAECEnv - The environment to wrap
        allow_flexible_task_tags: bool - allow task-agnostic actions to have invalid task channels
    Returns:
        BatchedAECEnv - The wrapped environment.
    """
    return shared_wrapper(env, ActionSpaceValidatorModifier, allow_flexible_task_tags=allow_flexible_task_tags)
