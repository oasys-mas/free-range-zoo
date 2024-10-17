from typing import Tuple

import torch
from torch import nn

from ..structures.state import WildfireState


class SuppressantRefillTransition(nn.Module):
    """
    Transition function for suppressant refills.

    Args:
        agent_shape: Tuple - Shape of the agent tensor
        stochastic_refill: bool - Whether to use stochastic refill
        refill_probability: float - Refill probability
        equipment_bonuses: torch.Tensor - Bonuses from equipment
    """

    def __init__(self, agent_shape: Tuple, stochastic_refill: bool, refill_probability: float, equipment_bonuses: torch.Tensor):
        """
        Initialize the transition object.

        Args:
            agent_shape: Tuple - Shape of the agent tensor (batches, agents)
            stochastic_refill: bool - Whether to have stochastic refills
            refill_probability: float - The probability of refill
            equipment_bonuses: torch.Tensor - The bonuses from equipment states
        """
        super().__init__()

        self.register_buffer('stochastic_refill', torch.tensor(stochastic_refill, dtype=torch.bool))
        self.register_buffer('refill_probability', torch.tensor(refill_probability, dtype=torch.float32))
        self.register_buffer('equipment_bonuses', equipment_bonuses)

        self.register_buffer('increase_mask', torch.zeros(agent_shape, dtype=torch.bool))

    def _reset_buffers(self):
        """Reset the transition buffers."""
        self.increase_mask[:, :] = False

    @torch.no_grad()
    def forward(
        self,
        state: WildfireState,
        refilled_suppressants: torch.Tensor,
        randomness_source: torch.Tensor,
        return_increased: bool = False,
    ) -> WildfireState:
        """
        Update the state of the suppressants.

        Args:
            state: WildfireState - Current state of the environment
            refilled_suppressants: torch.Tensor - Mask of agents which have successfully refilled their suppressants
            randomness_source: torch.Tensor - Randomness source
            return_increased: bool - Whether to return the increased suppressants
        """
        self._reset_buffers()

        self.increase_mask[refilled_suppressants] = True
        if self.stochastic_refill:
            self.increase_mask = torch.logical_and(self.increase_mask, randomness_source < self.refill_probability)

        equipment_states = state.equipment.flatten().unsqueeze(1)
        equipment_bonuses = self.equipment_bonuses[equipment_states].reshape(self.increase_mask.shape[0], -1)

        state.suppressants[self.increase_mask] = state.capacity[self.increase_mask]
        state.suppressants[self.increase_mask] += equipment_bonuses[self.increase_mask]

        if return_increased:
            return state, self.increase_mask

        return state
