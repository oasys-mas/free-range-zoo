"""Action Task Mapping Wrapper for Multi-Task Environments.

This module provides the ActionTaskMappingWrapperModifier class for mapping
actions to individual tasks in multi-task environments.
"""
from typing import Dict, Any, Tuple
from supersuit.generic_wrappers.utils.base_modifier import BaseModifier
import torch

from free_range_zoo.envs._base.v0.env import BatchedAECEnv
from free_range_zoo.wrappers.wrapper_util import shared_wrapper


class ActionTaskMappingWrapperModifier(BaseModifier):
    """Wrapper for mapping actions to tasks in a multi-task environment."""
    env = True
    subject_agent = True

    def __init__(self, env: BatchedAECEnv, subject_agent: str):
        """Initialize the ActionTaskMappingWrapperModifier.

        Args:
            env (BatchedAECEnv): The environment to wrap.
            subject_agent (str): The subject agent of the graph wrapper.

        Returns:
            None: None
        """
        self.env = env

        # Unpack the the parallel environment if it is wrapped in one.
        if hasattr(self.env, 'aec_env'):
            self.env = self.env.aec_env
        # Unpack the order enforcing wrapper if it has one of those.
        if hasattr(self.env, 'env'):
            self.env = self.env.env

        self.subject_agent = subject_agent

        self.cur_obs = None

    def modify_obs(self, observation: torch.Tensor) -> Tuple[Any, Dict[str, torch.IntTensor]]:
        """Modify the observation before it is passed to the agent.

        Args:
            observation (torch.Tensor): The observation to modify.

        Returns:
            Tuple[Any, Dict[str, torch.IntTensor]]: The original observation space followed by the mapping of actions to tasks.
        """
        self.cur_obs = observation, {'agent_action_mapping': self.env.agent_action_mapping[self.subject_agent]}
        return self.cur_obs


def action_mapping_wrapper_v0(env: BatchedAECEnv, **kwargs: Dict[str, Any]) -> BatchedAECEnv:
    """Apply the ActionTaskMappingWrapperModifier to the environment.

    Args:
        env (BatchedAECEnv): The environment to wrap.
        **kwargs (Dict[str, Any]): Additional keyword arguments passed to the modifier.

    Returns:
        BatchedAECEnv: The wrapped environment.
    """
    return shared_wrapper(env, ActionTaskMappingWrapperModifier, **kwargs)
