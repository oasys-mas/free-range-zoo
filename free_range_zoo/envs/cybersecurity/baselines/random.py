"""Agent the samples actions available to it in a uniform distribution."""
import torch
import free_range_rust
from free_range_zoo.utils.agent import Agent


class RandomBaseline(Agent):
    """Agent that samples actions avaialable to it in a uniform distribution."""

    def act(self, action_space: free_range_rust.Space) -> torch.IntTensor:
        """
        Return a list of actions, one for each parallel environment.

        Args:
            action_space: free_range_rust.Space - Current action space available to the agent.
        Returns:
            torch.IntTensor - List of actions, one for each parallel environment.
        """
        return torch.tensor(action_space.sample_nested(), dtype=torch.int32)
