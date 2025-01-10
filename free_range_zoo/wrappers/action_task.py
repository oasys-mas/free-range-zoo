"""Action task mapping wrapper for mapping actions to individual tasks in multi-task environments."""
from typing import Dict, Any, Tuple
from supersuit.generic_wrappers.utils.base_modifier import BaseModifier
import torch

from free_range_zoo.wrappers.wrapper_util import shared_wrapper
from free_range_zoo.utils.env import BatchedAECEnv


class ActionTaskMappingWrapperModifier(BaseModifier):
    """Wrapper for mapping actions to tasks in a multi-task environment."""
    env = True
    subject_agent = True

    def __init__(self, env: BatchedAECEnv, subject_agent: str, collapse: bool = False):
        """
        Initialize the ActionTaskMappingWrapperModifier.

        Args:
            env: BatchedAECEnv - The environment to wrap.
            subject_agent: str - The subject agent of the graph wrapper.
            collapse: bool - Whether to collapse the task-action and task-agnostic action nodes into single nodes.
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
        """
        Modify the observation before it is passed to the agent.

        Args:
            observation: The observation to modify.
        Returns:
            Tuple[Any, Dict[str, torch.IntTensor]]: The original observation space followed by the mapping of actions to tasks.
        """
        self.cur_obs = observation, {'agent_action_mapping': self.env.agent_action_mapping[self.subject_agent]}
        return self.cur_obs


def action_mapping_wrapper_v0(env, collapse: bool = False) -> BatchedAECEnv:
    """
    Apply the HeteroGraphWrapperModifier to the environment.

    Args:
        env: BatchedAECEnv - The environment to wrap.
        collapse: bool - Whether to collapse the task-action and task-agnostic action nodes into single nodes.
    Returns:
        BatchedAECEnv - The wrapped environment.
    """
    return shared_wrapper(env, ActionTaskMappingWrapperModifier, collapse=collapse)
