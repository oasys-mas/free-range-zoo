"""Agent that moves to the modulo of its agent index and then continually patches."""

from typing import List, Dict, Any
import torch
import free_range_rust
from free_range_zoo.utils.agent import Agent


class CampDefenderBaseline(Agent):
    """Agent that moves to the modulo of its agent index and then continually patches."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the agent."""
        super().__init__(*args, **kwargs)

        self.agent_index = int(self.agent_name.split('_')[-1])

        self.target_node = -1
        self.actions = torch.zeros((self.parallel_envs, 2), dtype=torch.int32)

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        """
        Return a list of actions, one for each parallel environment.

        Args:
            action_space: free_range_rust.Space - Current action space available to the agent.
        Returns:
            List[List[int]] - List of actions, one for each parallel environment.
        """
        return self.actions

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Observe the environment.

        Args:
            observation: Dict[str, Any] - Current observation from the environment.
        """
        self.observation, self.t_mapping = observation
        self.t_mapping = self.t_mapping['action_task_mappings']
        self.t_mapping = self.t_mapping.to_padded_tensor(padding=-100)

        self.target_node = self.agent_index % self.observation['tasks'].size(1)

        absent = self.observation['self'][:, 1] == 0
        location = self.observation['self'][:, 2]

        at_target_node = location == self.target_node

        # # Any agents that are targeted and not in location move
        self.actions[:, 0] = self.target_node
        self.actions[:, 1].masked_fill_(~at_target_node, 0)

        self.actions[:, 1].masked_fill_(absent, -1)  # Agents that are not present in the environment noop
        self.actions[:, 1].masked_fill_(at_target_node, -2)  # Any agents that are targeted and in location patch
