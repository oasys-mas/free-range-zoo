"""Camp defender and patcher baselines for cybersecurity environments.

This module provides baseline agents for cybersecurity environments that move to a target node (based on agent
index modulo the number of nodes) and continually patch that node. These agents demonstrate a simple, deterministic
defense strategy for benchmarking and comparison purposes.
"""
from typing import Any, Dict
import torch

import free_range_rust

from free_range_zoo.envs._base.agent import Agent


class CampDefenderBaseline(Agent):
    """Defender agent that moves to a target node and continually patches it.

    This agent selects a target node based on its agent index modulo the number of nodes, moves to that node,
    and repeatedly issues patch actions as long as it remains present. If the agent is absent, it performs a
    no-op. This baseline demonstrates a simple, deterministic defense strategy for cybersecurity environments.
    """

    def act(self, action_space: free_range_rust.Space) -> torch.IntTensor:
        """Select and return actions for each parallel environment.

        Args:
            action_space (free_range_rust.Space): The current action space available to the agent.

        Returns:
            torch.IntTensor: Tensor of actions, one for each parallel environment.
        """
        return self.actions

    def observe(self, observation: Dict[str, Any]) -> None:
        """Update internal state with the current environment observation.

        Args:
            observation (Dict[str, Any]): Current observation from the environment.
        """
        self.observation, self.t_mapping = observation
        self.t_mapping = self.t_mapping['action_task_mappings']

        if self.t_mapping.numel() == 0:
            self.actions[:, 0] = -100
            self.actions[:, 1] = -1
            return

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
