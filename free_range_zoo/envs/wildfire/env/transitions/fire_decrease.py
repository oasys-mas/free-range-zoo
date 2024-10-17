from typing import Tuple, Union
import torch
from torch import nn

from ..structures.state import WildfireState


class FireDecreaseTransition(nn.Module):
    """
    Transition function for the fire intensity decrease.

    Args:
        fire_shape: Tuple - Shape of the fire tensor
        stochastic_decrease: bool - Whether to use stochastic reduction
        decrease_probability: float - The base probability that a fire decreases in intensity once suppressant needs are met
        extra_power_decrease_bonus: float - The fire reduction per extra power applied to the fire
    """

    def __init__(self, fire_shape: Tuple, stochastic_decrease: bool, decrease_probability: float,
                 extra_power_decrease_bonus: float):
        super().__init__()

        self.register_buffer('stochastic_decrease', torch.tensor(stochastic_decrease, dtype=torch.bool))
        self.register_buffer('decrease_probability', torch.tensor(decrease_probability, dtype=torch.float32))
        self.register_buffer('extra_power_decrease_bonus', torch.tensor(extra_power_decrease_bonus, dtype=torch.float32))

        self.register_buffer('decrease_probabilities', torch.zeros(fire_shape, dtype=torch.float32))

    def _reset_buffers(self):
        """
        Reset the transition buffers
        """
        self.decrease_probabilities.fill_(0.0)

    @torch.no_grad()
    def forward(self,
                state: WildfireState,
                attack_counts: torch.Tensor,
                randomness_source: torch.Tensor,
                return_put_out: bool = False) -> Union[WildfireState, Tuple[WildfireState, torch.Tensor]]:
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
        suppressant_needs_met = torch.logical_and(attack_difference <= 0, lit_tiles)

        self.decrease_probabilities[:, :, :] = 0.0

        if self.stochastic_decrease:
            self.decrease_probabilities[suppressant_needs_met] = self.decrease_probability + \
                -1 * attack_difference[suppressant_needs_met] * self.extra_power_decrease_bonus
        else:
            self.decrease_probabilities[suppressant_needs_met] = 1.0

        self.decrease_probabilities = torch.clamp(self.decrease_probabilities, 0, 1)
        fire_decrease_mask = randomness_source < self.decrease_probabilities

        state.intensity[fire_decrease_mask] -= 1
        just_put_out = torch.logical_and(fire_decrease_mask, state.intensity <= 0)
        state.fires[just_put_out] *= -1
        state.fuel[just_put_out] -= 1

        if return_put_out:
            return state, just_put_out

        return state
