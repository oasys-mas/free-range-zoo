"""Agent that always performs a no-op action."""
from typing import List
import torch

import free_range_rust
from free_range_zoo.utils.agent import Agent


class NoopBaseline(Agent):
    """Agent that always performs a no-op action."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the agent."""
        super().__init__(*args, **kwargs)

        self.actions = torch.zeros((self.parallel_envs, 2), dtype=torch.int32)

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        """
        Return a list of actions, one for each parallel environment.

        Args:
            action_space: free_range_rust.Space - Current action space available to the agent.
        Returns:
            List[List[int]] - List of actions, one for each parallel environment.
        """
        self.actions[:, 1] = -1
        return self.actions
