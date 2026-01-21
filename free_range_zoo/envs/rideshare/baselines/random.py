"""Agent the samples actions available to it in a uniform distribution."""
from typing import List
import free_range_rust
from free_range_zoo.utils.agent import Agent
import torch


class RandomBaseline(Agent):
    """Agent that samples actions avaialable to it in a uniform distribution."""

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        """
        Return a list of actions, one for each parallel environment.

        Args:
            action_space: free_range_rust.Space - Current action space available to the agent.
        Returns:
            List[List[int]] - List of actions, one for each parallel environment.
        """
        seed = torch.randint(0, 2**64, (1, ), generator=self.generator, dtype=torch.uint64).item()
        return torch.tensor(action_space.sample_nested_with_seed(seed), dtype=torch.int16)
