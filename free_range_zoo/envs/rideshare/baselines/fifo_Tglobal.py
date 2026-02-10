"""
Global FIFO passenger selection baseline agent for the rideshare environment.

This module provides a baseline agent that always selects the globally earliest-arrived passenger for each agent in
every parallel environment. If multiple passengers share the earliest arrival, one is selected at random. The agent
performs pickup, dropoff, or accept actions as appropriate based on passenger state, serving as a deterministic
benchmark for rideshare task allocation.
"""
import free_range_rust
import torch

from free_range_zoo.envs._base.agent import Agent


class FirstInFirstOutTglobalBaseline(Agent):
    """
    Always select the first arrived passenger globally in the rideshare environment.

    This baseline agent always chooses the globally earliest-arrived passenger for each agent in every parallel
    environment. If multiple passengers share the earliest arrival, one is selected at random. The agent will
    perform pickup, dropoff, or accept actions as appropriate based on passenger state.
    """

    def act(self, action_space: free_range_rust.Space) -> torch.IntTensor:
        """Return a list of actions, one for each parallel environment.

        Args:
            action_space (free_range_rust.Space): Current action space available to the agent.

        Returns:
            torch.IntTensor: Tensor of actions, one for each parallel environment.
        """
        return self.actions

    def observe(self, observation: torch.Tensor) -> None:
        """Observe the current state of the environment.

        Args:
            observation (torch.Tensor): The observation from the environment.
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

        argmin_store = torch.ones_like(self.t_mapping) * float('inf')

        for batch in range(self.parallel_envs):
            for element in range(self.t_mapping[batch].size(0)):
                argmin_store[batch][element] = passengers[batch][element]

            if len(argmin_store[batch]) == 0:
                self.actions[batch].fill_(-1)
                continue

            max_val = argmin_store[batch].min()
            max_indices = torch.where(argmin_store[batch] == max_val)[0]

            if max_indices.shape[0] > 0:
                act = max_indices[torch.randint(0, max_indices.shape[0], (1, ), generator=self.generator)]
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
