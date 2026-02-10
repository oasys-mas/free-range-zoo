"""Generic Agent Interface for FreeRangeZoo Environments.

This module provides the abstract Agent base class that all agent
implementations must inherit from. Agents interact with environments
by observing states and selecting actions.

Classes:
    Agent: Abstract base class for all agent implementations.

Example Usage:
    >>> from free_range_zoo.envs._base.agent import Agent
    >>> class MyAgent(Agent):
    ...     def act(self, action_space):
    ...         # Return action tensor
    ...         pass
"""

from abc import ABC
from typing import List, Dict, Any

import torch

import free_range_rust


class Agent(ABC):
    """Generic interface for agents."""

    def __init__(self, agent_name: str, parallel_envs: int, seed: int = None) -> None:
        """
        Initialize the agent.

        Args:
            agent_name (str): Name of the subject agent
            parallel_envs (int): Number of parallel environments to operate on
            seed (Optional[int]): If provided, seed for internal torch.Generator (for reproducibility)
        """
        self.agent_name = agent_name
        self.parallel_envs = parallel_envs

        self.actions = torch.zeros((self.parallel_envs, 2), dtype=torch.int32)

        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        """
        Return a list of actions, one for each parallel environment.

        Args:
            action_space (free_range_rust.Space): Current action space available to the agent.

        Returns:
            List[List[int]]: List of actions, one for each parallel environment.
        """
        pass

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Observe the environment.

        Args:
            observation (Dict[str, Any]): Current observation from the environment.
        """
        pass
