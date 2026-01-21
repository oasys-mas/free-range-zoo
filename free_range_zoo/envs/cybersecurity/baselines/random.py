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
        seed = torch.randint(0, 2**64, (1, ), generator=self.generator, dtype=torch.uint64).item()
        return torch.tensor(action_space.sample_nested_with_seed(seed), dtype=torch.int32)
