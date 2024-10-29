import torch
from torch import nn

from ..structures.state import WildfireState


class EquipmentTransition(nn.Module):
    """
    Transition for calculating the modified equipment conditions
.
    Calculate the new modified equipment conditions
        - if the equipment is pristine, it can degrade or have a critical error
        - if the equipment is damaged, it can be repaired
        - if the equipment is in an intermediate state, it can degrade
    """

    def __init__(self, equipment_states: torch.Tensor, stochastic_repair: bool, repair_probability: float,
                 stochastic_degrade: bool, degrade_probability: float, critical_error: bool, critical_error_probability: float):
        """
        Initialize the transition object.

        Args:
            equipment_states: torch.Tensor - The definitions of different equipment conditions from the configuration
            stochastic_repair: bool - Whether to have a stochastic repair
            repair_probability: float - The probability of repair
            stochastic_degrade: bool - Whether to have a stochastic degrade
            degrade_probability: float - The probability of degradation
            critical_error: bool - Whether to have a critical error
            critical_error_probability: float - The probability of a critical error
        """
        super().__init__()

        self.register_buffer('equipment_states', equipment_states)
        self.register_buffer('stochastic_repair', torch.tensor(stochastic_repair, dtype=torch.bool))
        self.register_buffer('repair_probability', torch.tensor(repair_probability, dtype=torch.float32))
        self.register_buffer('stochastic_degrade', torch.tensor(stochastic_degrade, dtype=torch.bool))
        self.register_buffer('degrade_probability', torch.tensor(degrade_probability, dtype=torch.float32))
        self.register_buffer('critical_error', torch.tensor(critical_error, dtype=torch.bool))
        self.register_buffer('critical_error_probability', torch.tensor(critical_error_probability, dtype=torch.float32))

    @torch.no_grad()
    def forward(self, state: WildfireState, randomness_source: torch.Tensor) -> WildfireState:
        """
        Update the state of the equipment conditions.

        Args:
            state: WildfireState - The current state of the environment
            randomness_source: torch.Tensor - Randomness source
        Returns:
            WildfireState - The updated state of the environment
        """
        pristine = state.equipment == (self.equipment_states.shape[0] - 1)
        damaged = state.equipment == 0
        intermediate = torch.logical_and(torch.logical_not(pristine), torch.logical_not(damaged))

        if self.stochastic_repair:
            repairs = torch.logical_and(damaged, randomness_source < self.repair_probability)
        else:
            repairs = damaged

        state.equipment[repairs] = self.equipment_states.shape[0] - 1

        if self.critical_error:
            criticals = torch.logical_and(pristine, randomness_source < self.critical_error_probability)
            state.equipment[criticals] = 0

        if self.stochastic_degrade:
            degrades = torch.logical_and(torch.logical_or(pristine, intermediate), randomness_source < self.degrade_probability)
        else:
            degrades = torch.logical_or(intermediate, pristine)

        if self.critical_error:
            degrades = torch.logical_and(degrades, torch.logical_not(criticals))

        state.equipment[degrades] -= 1

        return state
