"""Random baseline policy for the wildfire environment."""
from typing import List
import free_range_rust
from free_range_zoo.utils.agent import Agent


class RandomBaseline(Agent):
    """Agent that samples actions avaialable to it in a uniform distribution."""

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        """
        Return a list of actions, one for each parallel environment.

        Args:
            action_space: free_range_rust.Space - Current action space available to the agent.
        Returns:
            List[List[int, int]] - List of actions, one for each parallel environment.
        """
        return action_space.sample_nested()
