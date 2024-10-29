from typing import Tuple, Union

import torch
from torch import nn

from ..structures.state import WildfireState


class FireIncreaseTransition(nn.Module):
    """
    Transition function for the fire intensity increase

    Args:
        fire_shape: Tuple - Shape of the fire tensor
        fire_states: int - The number of fire states
        stochastic_increase: bool - Whether to use stochastic increase in intensity
        intensity_increase_probability: float - The probability of fire intensity increase
        stochastic_burnouts: bool - Whether to have stochastic burnouts
        burnout_probability: float - The probability of burnout
    """

    def __init__(self, fire_shape: Tuple, fire_states: int, stochastic_increase: bool, intensity_increase_probability: float,
                 stochastic_burnouts: bool, burnout_probability: float):
        super().__init__()

        self.register_buffer('almost_burnout_state', torch.tensor(fire_states - 2, dtype=torch.int32))
        self.register_buffer('burnout_state', torch.tensor(fire_states - 1, dtype=torch.int32))

        self.register_buffer('stochastic_increase', torch.tensor(stochastic_increase, dtype=torch.bool))
        self.register_buffer('intensity_increase_probability', torch.tensor(intensity_increase_probability, dtype=torch.float32))
        self.register_buffer('stochastic_burnouts', torch.tensor(stochastic_burnouts, dtype=torch.bool))
        self.register_buffer('burnout_probability', torch.tensor(burnout_probability, dtype=torch.float32))

        self.register_buffer('increase_probabilities', torch.zeros(fire_shape, dtype=torch.float32))

    def _reset_buffers(self):
        """
        Reset the transition buffers
        """
        self.increase_probabilities.fill_(0.0)

    @torch.no_grad()
    def forward(self,
                state: WildfireState,
                attack_counts: torch.Tensor,
                randomness_source: torch.Tensor,
                return_burned_out: bool = False) -> Union[WildfireState, Tuple[WildfireState, torch.Tensor]]:
        """
        Update the state of the fire intensity

        Args:
            state: WildfireState - The current state of the environment
            attack_counts: torch.Tensor - The number of suppressants used on each cell
            randomness_source: torch.Tensor - Randomness source
            return_put_out: bool - Whether to return the put out fires

        Returns:
            WildfireState - The updated state of the environment
            torch.Tensor - A mask of the fires that were just put out
        """
        self._reset_buffers()

        required_suppressants = torch.where(state.fires >= 0, state.fires, torch.zeros_like(state.fires))
        attack_difference = required_suppressants - attack_counts

        lit_tiles = torch.logical_and(state.fires > 0, state.intensity > 0)
        suppressant_needs_unmet = torch.logical_and(attack_difference > 0, lit_tiles)

        almost_burnouts = torch.logical_and(suppressant_needs_unmet, state.intensity == self.almost_burnout_state)
        increasing = torch.logical_and(suppressant_needs_unmet, ~almost_burnouts)

        if self.stochastic_increase:
            self.increase_probabilities[increasing] = self.intensity_increase_probability
        else:
            self.increase_probabilities[increasing] = 1.0

        if self.stochastic_burnouts:
            self.increase_probabilities[almost_burnouts] = self.burnout_probability
        else:
            self.increase_probabilities[almost_burnouts] = self.intensity_increase_probability

        self.increase_probabilities = torch.clamp(self.increase_probabilities, 0, 1)
        fire_increase_mask = randomness_source < self.increase_probabilities

        state.intensity[fire_increase_mask] += 1

        just_burned_out = torch.logical_and(fire_increase_mask, state.intensity >= self.burnout_state)

        state.fires[just_burned_out] *= -1
        state.fuel[just_burned_out] = 0

        if return_burned_out:
            return state, just_burned_out

        return state
