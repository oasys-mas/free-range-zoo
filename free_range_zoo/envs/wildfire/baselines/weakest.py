"""
Weakest fire targeting baseline agent for wildfire environments.

This module provides a baseline agent that always targets the fire with the lowest intensity in the wildfire
environment. If multiple fires share the minimum strength, one is selected at random. This agent is useful for
benchmarking and as a deterministic control policy.
"""
from typing import Any, Dict

import free_range_rust
import torch

from free_range_zoo.utils.agent import Agent


class WeakestBaseline(Agent):
    """
    Target the weakest available fire in the wildfire environment.

    This baseline agent always selects the fire with the lowest intensity for each agent in every parallel
    environment. If multiple fires share the minimum strength, one is selected at random. Agents without suppressant
    perform a no-op action.
    """

    def act(self, action_space: free_range_rust.Space) -> torch.IntTensor:
        """
        Select and return actions for each parallel environment.

        Args:
            action_space: free_range_rust.Space - The current action space available to the agent.
        Returns:
            torch.IntTensor: Tensor of actions, one for each parallel environment.
        """
        return self.actions

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Update internal state with the current environment observation.

        Args:
            observation: Dict[str, Any] - The current observation from the environment.
        """
        self.observation, self.t_mapping = observation
        self.t_mapping = self.t_mapping['agent_action_mapping']

        has_suppressant = self.observation['self'][:, 3] != 0

        if self.t_mapping.numel() == 0:
            self.actions.fill_(-1)
            return

        fires = self.observation['tasks'].to_padded_tensor(-100)[:, :, 3]

        argmax_store = torch.empty_like(self.t_mapping)

        for batch in range(self.parallel_envs):
            for element in range(self.t_mapping[batch].size(0)):
                argmax_store[batch][element] = fires[batch][element]

            if len(argmax_store[batch]) == 0:
                self.actions[batch].fill_(-1)
                continue

            min_val = argmax_store[batch].min()
            min_indices = torch.where(argmax_store[batch] == min_val)[0]
            if min_indices.shape[0] > 0:
                act = min_indices[torch.randint(0, min_indices.shape[0], (1, ), generator=self.generator)]

            self.actions[batch, 0] = act
            self.actions[batch, 1] = 0

        self.actions[:, 1].masked_fill_(~has_suppressant, -1)  # Agents that do not have suppressant noop
