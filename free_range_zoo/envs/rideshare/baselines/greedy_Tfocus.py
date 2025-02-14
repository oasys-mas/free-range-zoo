"""Agent that always performs a no-op action."""
from typing import List
import torch

import free_range_rust
from free_range_zoo.utils.agent import Agent
from free_range_zoo.envs.rideshare.env.transitions.movement import MovementTransition
from free_range_zoo.envs.rideshare.env.structures.configuration import AgentConfiguration


class GreedyTaskFocus(Agent):
    """Agent that acts on the soonest to completion passenger. i.e. minimum total distance to completion. (works on a task until completion)"""

    def __init__(self, *args, agent_configuration: AgentConfiguration, **kwargs) -> None:
        """Initialize the agent."""
        super().__init__(*args, **kwargs)

        self.actions = torch.zeros((self.parallel_envs, 2), dtype=torch.int32)
        self.use_diagonal_travel = agent_configuration.use_diagonal_travel

        #used for distance calculations
        self.movement_transition = MovementTransition(
            parallel_envs=self.parallel_envs,
            num_agents=1,
            fast_travel=True,
            diagonal_travel=self.use_diagonal_travel,
        )

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

        #no passengers, (all accepted by other agents)
        if all([self.observation['tasks'][i].size(0) == 0 for i in range(self.parallel_envs)]):
            self.actions.fill_(-1)
            return

        accepted = self.observation['tasks'].to_padded_tensor(-100)[:, :, 4] > 0
        riding = self.observation['tasks'].to_padded_tensor(-100)[:, :, 5] > 0
        unaccepted = ~accepted & ~riding

        passenger_current = self.observation['tasks'].to_padded_tensor(-100)[:, :, [0, 1]]
        passenger_destination = self.observation['tasks'].to_padded_tensor(-100)[:, :, [2, 3]]
        my_location = self.observation['self'][:, [0, 1]].unsqueeze(1).repeat(1, passenger_current.size(1), 1)

        _, passenger_distance = self.movement_transition.distance(starts=passenger_current, goals=passenger_destination)
        _, my_distance = self.movement_transition.distance(starts=my_location, goals=passenger_current)
        passengers = passenger_distance + my_distance

        argmin_store = torch.empty_like(self.t_mapping)

        for batch in range(self.parallel_envs):

            #only one task at a time filtering <prevents acceptance of tasks when one is already being worked on>
            if torch.any(accepted[batch]):
                passengers[batch][~accepted[batch]] = float('inf')
                assert (passengers[batch][accepted[batch]] < float('inf')).sum() == 1,\
                    "Invalid Observation, if this is reached there exists >=1 passenger, but there should only be one accepted passenger"+\
                        f"\n{passengers[batch][accepted[batch]][(passengers[batch][accepted[batch]] < float('inf'))]}"

            for element in range(self.t_mapping[batch].size(0)):
                argmin_store[batch][element] = passengers[batch][element]

            if len(argmin_store[batch]) == 0:
                self.actions[batch].fill_(-1)  # There are no passengers seen in the environment so this agent (batch) must noop
                continue

            self.actions[batch, 0] = argmin_store[batch].argmin(dim=0)

            #dropoff
            if riding[batch][self.actions[batch, 0]]:
                self.actions[batch, 1] = 2

            #pickup
            elif accepted[batch][self.actions[batch, 0]]:
                self.actions[batch, 1] = 1

            #accept
            elif unaccepted[batch][self.actions[batch, 0]]:
                self.actions[batch, 1] = 0

            #noop
            else:
                raise ValueError(
                    "Invalid Observation, if this is reached there exists >=1 passenger, but that passenger has no features")
