import torch


def calculate_fire_increase_probabilities(fires: torch.Tensor,
                                          intensity: torch.Tensor,
                                          attack_counts: torch.Tensor,
                                          use_almost_burned_out: bool,
                                          almost_burned_out: int,
                                          intensity_increase_probability: float,
                                          burnout_probability: float,
                                          device: torch.DeviceObjType = torch.device('cpu')) -> torch.Tensor:
    """
    Calculate the probability of fire intensity increase

    Args:
        fires: torch.Tensor - The fire intensity of each cell
        intensity: torch.Tensor - The intensity of each fire
        attack_counts: torch.Tensor - The number of suppressants used on each cell
        use_almost_burned_out: bool - Whether to use almost burned out
        almost_burned_out: int - The intensity value for almost burned out
        intensity_increase_probability: float - The probability of fire intensity increase
        burnout_probability: float - The probability of burnout
        device: torch.DeviceObjType - Device to move tensors to

    Returns:
        torch.Tensor - The probability of fire intensity increase
    """
    required_suppressants = torch.where(fires >= 0, fires, torch.zeros_like(fires))
    attack_difference = required_suppressants - attack_counts

    lit_tiles = torch.logical_and(fires > 0, intensity > 0)
    suppressant_needs_unmet = torch.logical_and(attack_difference > 0, lit_tiles)

    fire_increase_probabilities = torch.zeros_like(attack_difference, device=device, dtype=torch.float)
    almost_burnouts = torch.logical_and(suppressant_needs_unmet, intensity == almost_burned_out)
    increasing = torch.logical_and(suppressant_needs_unmet, ~almost_burnouts)

    if use_almost_burned_out:
        fire_increase_probabilities[increasing] = intensity_increase_probability
        fire_increase_probabilities[almost_burnouts] = burnout_probability
    else:
        all_increasing = torch.logical_or(increasing, almost_burnouts)
        fire_increase_probabilities[all_increasing] = intensity_increase_probability

    return torch.clamp(fire_increase_probabilities, 0, 1)
