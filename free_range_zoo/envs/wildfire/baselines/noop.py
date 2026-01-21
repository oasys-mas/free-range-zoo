"""
No-op baseline agent for the wildfire environment.

This module provides a baseline agent that always performs a no-op action for every agent in every parallel
environment, regardless of the state or available actions. This agent is useful for benchmarking, debugging,
or as a control policy to compare against more sophisticated wildfire suppression strategies.
"""
import free_range_rust
import torch

from free_range_zoo.utils.agent import Agent


class NoopBaseline(Agent):
    """
    Always perform a no-op action in the wildfire environment.

    This baseline agent always returns a no-op action for every agent in every parallel environment, regardless
    of the state or available actions. Useful for benchmarking or as a control policy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.actions[:, 0] = 0
        self.actions[:, 1] = -1

    def act(self, action_space: free_range_rust.Space) -> torch.IntTensor:
        """
        Return a list of no-op actions for each parallel environment.

        Args:
            action_space: free_range_rust.Space - The current action space available to the agent.
        Returns:
            torch.IntTensor: Tensor of actions, one for each parallel environment.
        """
        return self.actions
