from typing import Tuple

import torch
from torch import nn

from ..structures.state import WildfireState


class SuppressantDecreaseTransition(nn.Module):
    """
    Transition function for suppressant decrease

    Args:
        agent_shape: Tuple - Shape of the agent tensor
        stochastic_decrease: bool - Whether to use stochastic decrease
        decrease_probability: float - Decrease probability
    """

    def __init__(self, agent_shape: Tuple, stochastic_decrease: bool, decrease_probability: float):
        super().__init__()

        self.register_buffer('stochastic_decrease', torch.tensor(stochastic_decrease, dtype=torch.bool))
        self.register_buffer('decrease_probability', torch.tensor(decrease_probability, dtype=torch.float32))

        self.register_buffer('decrease_mask', torch.zeros(agent_shape, dtype=torch.bool))

    def _reset_buffers(self):
        """
        Reset the transition buffers
        """
        self.decrease_mask.fill_(False)

    @torch.no_grad()
    def forward(self,
                state: WildfireState,
                used_suppressants: torch.Tensor,
                randomness_source: torch.Tensor,
                return_decreased: bool = False) -> WildfireState:
        """
        Update the suppressants tensor

        Args:
            state: WildfireState - The current state of the environment
            used_suppressants: torch.Tensor - Locations of agents that used suppressants
            randomness_source: torch.Tensor - Randomness source
            return_decreased: bool - Whether to return a mask of the agents that decreased suppressants
        Returns:
            WildfireState - The updated state of the environment with suppressants decreased
            torch.Tensor - A mask of the agents that decreased in suppresants
        """
        self._reset_buffers()

        self.decrease_mask = used_suppressants
        if self.stochastic_decrease:
            self.decrease_mask = torch.logical_and(self.decrease_mask, randomness_source < self.decrease_probability)

        state.suppressants = torch.where(self.decrease_mask, state.suppressants - 1, state.suppressants)
        state.suppressants = torch.clamp(state.suppressants, min=0)

        if return_decreased:
            return state, self.decrease_mask

        return state
