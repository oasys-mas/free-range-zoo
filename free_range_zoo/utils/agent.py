"""Generic interface for agents."""
from abc import ABC

from typing import List, Dict, Any
import free_range_rust


class Agent(ABC):
    """Generic interface for agents."""

    def __init__(self, agent_name: str, parallel_envs: int) -> None:
        """
        Initialize the agent.

        Args:
            agent_name: str - Name of the subject agent
            parallel_envs: int - Number of parallel environments to operate on
        """
        self.agent_name = agent_name
        self.parallel_envs = parallel_envs

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
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
