import torch
from torch import nn

from ..structures.state import WildfireState


class EquipmentTransition(nn.Module):
    """
    Transition for calculating the modified equipment conditions.

    Calculate the new modified equipment conditions
        - if the equipment is pristine, it can degrade or have a critical error
        - if the equipment is damaged, it can be repaired
        - if the equipment is in an intermediate state, it can degrade

    When condition_on_actions is True:
        - Degradation only occurs when the agent fought
        - Repair only occurs when the agent noops at fully damaged
        - Critical errors only occur when the agent fought at pristine
        This is implemented by zeroing out probabilities for incorrect action types.
    """

    def __init__(self,
                 equipment_states: torch.Tensor,
                 stochastic_repair: bool,
                 repair_probability: float,
                 stochastic_degrade: bool,
                 degrade_probability: float,
                 critical_error: bool,
                 critical_error_probability: float,
                 condition_on_actions: bool = False):
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
            condition_on_actions: bool - Whether equipment transitions are conditioned on agent actions
        """
        super().__init__()

        self.register_buffer('equipment_states', equipment_states)
        self.register_buffer('stochastic_repair', torch.tensor(stochastic_repair, dtype=torch.bool))
        self.register_buffer('repair_probability', torch.tensor(repair_probability, dtype=torch.float32))
        self.register_buffer('stochastic_degrade', torch.tensor(stochastic_degrade, dtype=torch.bool))
        self.register_buffer('degrade_probability', torch.tensor(degrade_probability, dtype=torch.float32))
        self.register_buffer('critical_error', torch.tensor(critical_error, dtype=torch.bool))
        self.register_buffer('critical_error_probability', torch.tensor(critical_error_probability, dtype=torch.float32))
        self.register_buffer('condition_on_actions', torch.tensor(condition_on_actions, dtype=torch.bool))

    @torch.no_grad()
    def forward(self,
                state: WildfireState,
                randomness_source: torch.FloatTensor,
                fought: torch.BoolTensor = None) -> WildfireState:
        """
        Update the state of the equipment conditions.

        Args:
            state: WildfireState - The current state of the environment
            randomness_source: torch.FloatTensor - Randomness source
            fought: torch.BoolTensor - tensor indicating which agents fought this step
        Returns:
            WildfireState - The updated state of the environment
        """
        pristine = state.equipment == (self.equipment_states.shape[0] - 1)
        damaged = state.equipment == 0
        intermediate = torch.logical_and(torch.logical_not(pristine), torch.logical_not(damaged))

        if fought is None and self.condition_on_actions:
            raise ValueError("fought tensor must be provided when condition_on_actions is True.")

        # Effective probabilities (zeroed out based on actions when condition_on_actions=True)
        effective_repair_prob = self.repair_probability
        effective_degrade_prob = self.degrade_probability
        effective_critical_prob = self.critical_error_probability

        if self.condition_on_actions:
            # Repair: only if damaged AND nooped (didn't fight)
            repair_mask = torch.logical_and(damaged, torch.logical_not(fought))
            effective_repair_prob = effective_repair_prob * repair_mask.to(torch.float32)

            # Degrade: only if (pristine OR intermediate) AND fought
            degrade_mask = torch.logical_and(torch.logical_or(pristine, intermediate), fought)
            effective_degrade_prob = effective_degrade_prob * degrade_mask.to(torch.float32)

            # Critical error: only if pristine AND fought
            critical_mask = torch.logical_and(pristine, fought)
            effective_critical_prob = effective_critical_prob * critical_mask.to(torch.float32)

        # Apply repairs
        if self.stochastic_repair:
            repairs = torch.logical_and(damaged, randomness_source < effective_repair_prob)
        else:
            repairs = damaged

        state.equipment[repairs] = self.equipment_states.shape[0] - 1

        # Apply critical errors
        if self.critical_error:
            criticals = torch.logical_and(pristine, randomness_source < effective_critical_prob)
            state.equipment[criticals] = 0

        # Apply degradation
        if self.stochastic_degrade:
            degrades = torch.logical_and(torch.logical_or(pristine, intermediate), randomness_source < effective_degrade_prob)
        else:
            degrades = torch.logical_or(intermediate, pristine)

        if self.critical_error:
            degrades = torch.logical_and(degrades, torch.logical_not(criticals))

        state.equipment[degrades] -= 1

        return state
