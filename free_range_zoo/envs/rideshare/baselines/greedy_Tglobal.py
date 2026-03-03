"""
Greedy global task selection baseline agent for the rideshare environment.

This module provides a baseline agent that always selects the passenger with the soonest completion globally for each agent in every parallel environment. If multiple passengers share the minimum, one is selected at random. The agent works on a task until completion, performing pickup, dropoff, or accept actions as appropriate, and serves as a deterministic benchmark for rideshare task allocation.
"""
import torch

import free_range_rust

from free_range_zoo.envs._base.agent import Agent
from free_range_zoo.envs.rideshare.env.v0.transitions.movement import MovementTransition
from free_range_zoo.envs.rideshare.env.v0.structures.configuration import AgentConfiguration


class GreedyTaskGlobal(Agent):
    """
    Always select the passenger with the soonest completion globally.

    This baseline agent chooses the globally minimum total distance to completion (pickup plus dropoff) for each
    agent in every parallel environment. If multiple passengers share the minimum, one is selected at random. The
    agent will work on a task until completion, performing pickup, dropoff, or accept actions as appropriate.
    """

    def __init__(self, *args, agent_configuration: AgentConfiguration, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_diagonal_travel = agent_configuration.use_diagonal_travel
        self.movement_transition = MovementTransition(
            parallel_envs=self.parallel_envs,
            num_agents=1,
            fast_travel=True,
            diagonal_travel=self.use_diagonal_travel,
        )

    def act(self, action_space: free_range_rust.Space) -> torch.IntTensor:
        """
        Select and return actions for each parallel environment.

        Args:
            action_space: free_range_rust.Space - The current action space available to the agent.
        Returns:
            torch.IntTensor: Tensor of actions, one for each parallel environment.
        """
        return self.actions

    def observe(self, observation: torch.Tensor) -> None:
        """
        Update internal state with the current environment observation.

        Args:
            observation: torch.Tensor - The observation from the environment.
        """
        self.observation, self.t_mapping = observation
        self.t_mapping = self.t_mapping['agent_action_mapping']

        # no passengers, (all accepted by other agents)
        if all([self.observation['tasks'][i].size(0) == 0 for i in range(self.parallel_envs)]):
            self.actions.fill_(-1)
            return

        accepted = self.observation['tasks'].to_padded_tensor(-100)[:, :, 4] >= 0
        riding = self.observation['tasks'].to_padded_tensor(-100)[:, :, 5] >= 0
        unaccepted = ~accepted & ~riding

        passenger_current = self.observation['tasks'].to_padded_tensor(-100)[:, :, [0, 1]]
        passenger_destination = self.observation['tasks'].to_padded_tensor(-100)[:, :, [2, 3]]
        my_location = self.observation['self'][:, [0, 1]].unsqueeze(1).repeat(1, passenger_current.size(1), 1)

        _, passenger_distance = self.movement_transition.distance(starts=passenger_current, goals=passenger_destination)
        _, my_distance = self.movement_transition.distance(starts=my_location, goals=passenger_current)
        passengers = passenger_distance + my_distance

        argmin_store = torch.empty_like(self.t_mapping)

        for batch in range(self.parallel_envs):

            for element in range(self.t_mapping[batch].size(0)):
                argmin_store[batch][element] = passengers[batch][element]

            if len(argmin_store[batch]) == 0:
                self.actions[batch].fill_(-1)  # There are no passengers seen in the environment so this agent (batch) must noop
                continue

            min_val = argmin_store[batch].min()
            min_indices = torch.where(argmin_store[batch] == min_val)[0]
            if min_indices.shape[0] > 0:
                act = min_indices[torch.randint(0, min_indices.shape[0], (1, ), generator=self.generator)]
            self.actions[batch, 0] = act

            # dropoff
            if riding[batch][self.actions[batch, 0]]:
                self.actions[batch, 1] = 2

            # pickup
            elif accepted[batch][self.actions[batch, 0]]:
                self.actions[batch, 1] = 1

            # accept
            elif unaccepted[batch][self.actions[batch, 0]]:
                self.actions[batch, 1] = 0

            # noop
            else:
                raise ValueError(
                    "Invalid Observation, if this is reached there exists >=1 passenger, but that passenger has no features")
