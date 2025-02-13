"""Agent that always fights the weakest available fire."""
from typing import List, Dict, Any
import torch
import free_range_rust
from free_range_zoo.utils.agent import Agent


class WeakestBaseline(Agent):
    """Agent that always fights the weakest available fire."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the agent."""
        super().__init__(*args, **kwargs)

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

        has_suppressant = self.observation['self'][:, 3] != 0
        fires = self.observation['tasks'].to_padded_tensor(-100)[:, :, 3]

        argmax_store = torch.empty_like(self.t_mapping)

        for batch in range(self.parallel_envs):
            for element in range(self.t_mapping[batch].size(0)):
                argmax_store[batch][element] = fires[batch][element]

            if len(argmax_store[batch]) == 0:
                self.actions[batch].fill_(-1)
                continue

            self.actions[batch, 0] = argmax_store[batch].argmin(dim=0)
            self.actions[batch, 1] = 0

        self.actions[:, 1].masked_fill_(~has_suppressant, -1)  # Agents that do not have suppressant noop
