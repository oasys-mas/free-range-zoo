import torch
import torch.nn.functional as F


def calculate_fire_spread_probabilities(fire_spread_weights: torch.Tensor,
                                        fires: torch.Tensor,
                                        intensities: torch.Tensor,
                                        fuel: torch.Tensor,
                                        use_fire_fuel: bool) -> torch.Tensor:
    """
    Calculates the probability of fire spreading to each cell using the spreading model from Eck et. al 2020

    Args:
        fire_spread_weights: torch.Tensor - The fire spread filter
        fires: torch.Tensor - The fire intensity of each cell
        intensities: torch.Tensor - The intensity of each fire
        fuel: torch.Tensor - The fuel remaining in each cell
        use_fire_fuel: bool - Whether to use fire fuel

    Returns:
        torch.Tensor - The probability of fire spreading to each cell
    """
    x = torch.where(fires != 0, 1, 0)
    x = F.pad(x, (1, 1, 1, 1)).to(torch.float32).unsqueeze(1)
    fire_spread_probabilities = F.conv2d(x, fire_spread_weights).squeeze(1)

    unlit_tiles = torch.logical_and(fires < 0, intensities == 0)
    if use_fire_fuel:
        unlit_tiles = torch.logical_and(unlit_tiles, fuel > 0)

    fire_spread_probabilities[~unlit_tiles] = 0

    return fire_spread_probabilities
