"""Agent that always performs a no-op action."""
from typing import List
import free_range_rust
from free_range_zoo.utils.agent import Agent


class NoopBaseline(Agent):
    """Agent that always performs a no-op action."""

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        """
        Return a list of actions, one for each parallel environment.

        Args:
            action_space: free_range_rust.Space - Current action space available to the agent.
        Returns:
            List[List[int]] - List of actions, one for each parallel environment.
        """
        actions = []
        for space in action_space.spaces:
            actions.append([0, -1])

        return actions
