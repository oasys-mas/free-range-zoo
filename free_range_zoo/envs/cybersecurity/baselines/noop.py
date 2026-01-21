"""No-op baseline agent for cybersecurity environments.

This module provides a baseline agent that always performs a no-op action, serving as a control or lower-bound
benchmark for agent performance in cybersecurity tasks.
"""
import free_range_rust
import torch

from free_range_zoo.utils.agent import Agent


class NoopBaseline(Agent):
    """
    Agent that always performs a no-op action.

    This agent issues a no-op action at every step, regardless of the environment state. It is useful as a control
    or lower-bound baseline for evaluating the effectiveness of more sophisticated cybersecurity agents.
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
