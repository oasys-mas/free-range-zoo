"""Agent that picks the weakest available fire and focuses on it until its out."""
from typing import List, Dict, Any
import free_range_rust
from free_range_zoo.utils.agent import Agent


class StrongestBaseline(Agent):
    """Agent that picks the strongest available fire and focuses on it until its out."""

    def act(self, action_space: free_range_rust.Space) -> List[List[int, int]]:
        """
        Return a list of actions, one for each parallel environment.

        Args:
            action_space: free_range_rust.Space - Current action space available to the agent.
        Returns:
            List[List[int, int]] - List of actions, one for each parallel environment.
        """
        pass

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Observe the environment.

        Args:
            observation: Dict[str, Any] - Current observation from the environment.
        """
        pass
