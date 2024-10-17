import torch

from torch import nn

from ..structures.state import WildfireState


class FireSpreadTransition(nn.Module):
    """
    Transition function for the fire spread model from Eck et al. 2020.

    Args:
        fire_spread_weights: torch.Tensor - The fire spread filter weights for each cell and neighbour
        ignition_temperatures: torch.Tensor - The ignition temperature for each cell
        use_fire_fuel: bool - Whether to use fire fuel
    """

    def __init__(self, fire_spread_weights: torch.Tensor, ignition_temperatures: torch.Tensor, use_fire_fuel: bool):
        super().__init__()

        self.register_buffer('ignition_temperatures', ignition_temperatures)
        self.register_buffer('use_fire_fuel', torch.tensor(use_fire_fuel, dtype=torch.bool))

        self.fire_spread_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.fire_spread_filter.weight.data = fire_spread_weights

    @torch.no_grad()
    def forward(self, state: WildfireState, randomness_source: torch.Tensor) -> WildfireState:
        """
        Update the state of the fire intensity.

        Args:
            state: WildfireState - The current state of the environment
            randomness_source: torch.Tensor - Randomness source
        Returns:
            WildfireState - The updated state of the environment
        """
        lit = torch.logical_and(state.fires > 0, state.intensity > 0)
        lit = lit.to(torch.float32).unsqueeze(1)
        fire_spread_probabilities = self.fire_spread_filter(lit).squeeze(1)

        unlit_tiles = torch.logical_and(state.fires < 0, state.intensity == 0)
        if self.use_fire_fuel:
            unlit_tiles = torch.logical_and(unlit_tiles, state.fuel > 0)

        fire_spread_probabilities[~unlit_tiles] = 0
        fire_spread_mask = randomness_source < fire_spread_probabilities

        state.fires[fire_spread_mask] *= -1
        state.intensity[fire_spread_mask] = self.ignition_temperatures.expand(state.fires.shape[0], -1, -1)[fire_spread_mask]

        return state
