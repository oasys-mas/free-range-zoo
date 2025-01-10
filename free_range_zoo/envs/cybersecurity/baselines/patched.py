"""Agent that picks the most patched node and continues to patch / attack it, repositioning every three timesteps."""

from typing import List, Dict, Any
import torch
import free_range_rust
from free_range_zoo.utils.agent import Agent


class PatchedAttackerBaseline(Agent):
    """Agent that picks the most patched node and attacks it for three consecutive steps."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the agent."""
        super().__init__(*args, **kwargs)

        self.target_node = torch.ones((self.parallel_envs, ), dtype=torch.int32) * -1
        self.time_focused = torch.zeros((self.parallel_envs, ), dtype=torch.int32)
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
        self.t_mapping = self.t_mapping['agent_action_mapping']
        self.t_mapping = self.t_mapping.to_padded_tensor(padding=-100)

        new_targets = self.observation['tasks'].argmin(dim=1).flatten()
        self.target_node = torch.where(self.target_node == -1, new_targets, self.target_node)

        absent = self.observation['self'][:, 1] == 0
        targeted = (self.target_node != -1) & ~absent

        # # Any agents that are targeted attack their target
        self.actions[:, 0] = self.target_node
        self.actions[:, 1].masked_fill_(targeted, 0)

        self.actions[:, 1].masked_fill_(absent, -1)  # Agents that are not present in the environment noop

        self.time_focused[targeted] += 1  # Increment the patch counter each time they patch

        # If the time focused is at 3, then unset the current target
        self.target_node = torch.where(self.time_focused >= 3, -1, self.target_node)
        self.time_focused = torch.where(self.time_focused >= 3, 0, self.time_focused)


class PatchedDefenderBaseline(Agent):
    """Agent that picks the most patched node and patches it for three consecutive steps."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the agent."""
        super().__init__(*args, **kwargs)

        self.target_node = torch.ones((self.parallel_envs, ), dtype=torch.int32) * -1
        self.time_focused = torch.zeros((self.parallel_envs, ), dtype=torch.int32)
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
        self.t_mapping = self.t_mapping['agent_action_mapping']
        self.t_mapping = self.t_mapping.to_padded_tensor(padding=-100)

        last_monitor = (self.observation['tasks'] != -100).all(dim=1).flatten()

        task_mask = self.observation['tasks']
        task_mask = torch.where(self.observation['tasks'] == 0, 1000, task_mask)
        task_mask = torch.where(self.observation['tasks'] == -100, 1000, task_mask)
        new_targets = task_mask.argmin(dim=1).flatten()

        self.target_node = torch.where(last_monitor, new_targets, self.target_node)

        absent = self.observation['self'][:, 1] == 0
        targeted = (self.target_node != -1) & ~absent
        targetless = (self.target_node == -1) & ~absent

        location = self.observation['self'][:, 2]
        at_target_node = location == self.target_node

        # # Any agents that are targeted and not in location move
        self.actions[:, 0] = self.target_node
        self.actions[:, 1].masked_fill_(targeted & ~at_target_node, 0)

        self.actions[:, 1].masked_fill_(absent, -1)  # Agents that are not present in the environment noop
        self.actions[:, 1].masked_fill_(targeted & at_target_node, -2)  # Any agents that are targeted and in location patch
        self.actions[:, 1].masked_fill_(targetless & ~last_monitor, -3)  # Agents that are targetless with no observation monitor

        # Increment the patch counter each time they patch
        self.time_focused[targeted & at_target_node] += 1

        # If the time focused is at 3, then unset the current target
        self.target_node = torch.where(self.time_focused >= 3, -1, self.target_node)
        self.time_focused = torch.where(self.time_focused >= 3, 0, self.time_focused)
