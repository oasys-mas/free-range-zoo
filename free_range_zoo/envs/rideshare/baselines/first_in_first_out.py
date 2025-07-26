"""Agent that always performs a no-op action."""
from typing import List
import torch

import free_range_rust
from free_range_zoo.utils.agent import Agent


class FirstInFirstOutBaseline(Agent):
    """Agent that always acts on the first arrived passenger."""

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

    def observe(self, observation: torch.Tensor) -> None:
        """
        Observe the current state of the environment.

        Args:
            observation: torch.Tensor - The observation from the environment.
        """
        self.observation, self.t_mapping = observation
        self.t_mapping = self.t_mapping['agent_action_mapping']

        if all([self.observation['tasks'][i].size(0) == 0 for i in range(self.parallel_envs)]):
            self.actions.fill_(-1)
            return

        passengers = self.observation['tasks'].to_padded_tensor(-100)[:, :, -1]
        accepted = self.observation['tasks'].to_padded_tensor(-100)[:, :, 4] >= 0
        riding = self.observation['tasks'].to_padded_tensor(-100)[:, :, 5] >= 0
        unaccepted = ~accepted & ~riding

        argmin_store = torch.empty_like(self.t_mapping)

        for batch in range(self.parallel_envs):
            for element in range(self.t_mapping[batch].size(0)):
                argmin_store[batch][element] = passengers[batch][element]

            if len(argmin_store[batch]) == 0:
                self.actions[batch].fill_(-1)
                continue

            max_val = argmin_store[batch].max()
            max_indices = torch.where(argmin_store[batch] == max_val)[0]
            if max_indices.shape[0] > 0:
                act = max_indices[torch.randint(0, max_indices.shape[0], (1, ))]
            self.actions[batch, 0] = act

            if riding[batch][self.actions[batch, 0]]:
                self.actions[batch, 1] = 2
            elif accepted[batch][self.actions[batch, 0]]:
                self.actions[batch, 1] = 1
            elif unaccepted[batch][self.actions[batch, 0]]:
                self.actions[batch, 1] = 0
            else:
                raise ValueError(
                    "Invalid Observation, if this is reached there exists >=1 passenger, but that passenger has no features")
