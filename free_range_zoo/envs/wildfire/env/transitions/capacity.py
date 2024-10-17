from typing import Tuple

import torch
from torch import nn

from ..structures.state import WildfireState


class CapacityTransition(nn.Module):
    """
    Transition function for the suppressant capacity

    Parameters:
        agent_shape: Tuple - Shape of the agent tensor
        stochastic_switch: bool - Whether to stochastically switch the maximum suppressant value
        tank_switch_probability: float - Probability of switching the maximum suppressant value
        possible_suppressant_maximums: torch.Tensor - Possible maximum suppressant values
        suppressant_maximum_probabilities: torch.Tensor - Probabilities of each maximum suppressant value
    """

    def __init__(self, agent_shape: Tuple, stochastic_switch: bool, tank_switch_probability: float,
                 possible_capacities: torch.Tensor, capacity_probabilities: torch.Tensor):
        super().__init__()

        self.register_buffer('stochastic_switch', torch.tensor(stochastic_switch, dtype=torch.bool))
        self.register_buffer('tank_switch_probability', torch.tensor(tank_switch_probability, dtype=torch.float32))
        self.register_buffer('possible_capacities', possible_capacities)
        self.register_buffer('capacity_probabilities', torch.cumsum(capacity_probabilities, dim=0))

        self.register_buffer('tank_switches', torch.zeros(agent_shape, dtype=torch.bool))

    def _reset_buffers(self) -> None:
        """
        Reset the transition buffers
        """
        self.tank_switches.fill_(False)

    @torch.no_grad()
    def forward(self, state: WildfireState, targets: torch.Tensor, randomness_source: torch.Tensor) -> WildfireState:
        """
        Update the state of the maximum suppressant capacity

        Args:
            state: WildfireState - Current state of the environment
            targets: torch.Tensor - Mask of agents which have successfully refilled their suppressants
            randomness_source: torch.Tensor - Randomness source
        Returns:
            WildfireState - Updated state of the environment
        """
        self._reset_buffers()

        size_indices = torch.bucketize(randomness_source[0], self.capacity_probabilities)
        new_maximums = self.possible_capacities[size_indices]

        if self.stochastic_switch:
            self.tank_switches[targets] = randomness_source[1][targets] < self.tank_switch_probability
        else:
            self.tank_switches[targets] = True

        bonuses = state.suppressants - state.capacity

        state.capacity[self.tank_switches] = new_maximums[self.tank_switches]
        state.suppressants[self.tank_switches] = new_maximums[self.tank_switches]
        state.suppressants[self.tank_switches] += bonuses[self.tank_switches]

        return state
