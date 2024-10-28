import torch


def calculate_fire_decrease_probabilities(fires: torch.Tensor,
                                          intensity: torch.Tensor,
                                          attack_counts: torch.Tensor,
                                          base_fire_reduction: float,
                                          fire_reduction_per_extra_agent: float,
                                          device: torch.DeviceObjType = torch.device('cpu')) -> torch.Tensor:
    """
    Calculate the probability of fire intensity decrease

    Args:
        fires: torch.Tensor - The fire intensity of each cell
        intensity: torch.Tensor - The intensity of each fire
        attack_counts: torch.Tensor - The number of suppressants used on each cell
        base_fire_reduction: float - The base fire reduction
        fire_reduction_per_extra_agent: float - The fire reduction per extra agent
        device: torch.DeviceObjType - Device to move tensors to

    Returns:
        torch.Tensor - The probability of fire intensity decrease
    """
    required_suppressants = torch.where(fires >= 0, fires, torch.zeros_like(fires))
    attack_difference = required_suppressants - attack_counts

    lit_tiles = torch.logical_and(fires > 0, intensity > 0)
    suppressant_needs_met = torch.logical_and(attack_difference <= 0, lit_tiles)

    fire_decrease_probabilities = torch.zeros_like(attack_difference, device=device, dtype=torch.float32)

    probabilities = base_fire_reduction + -1 * attack_difference * fire_reduction_per_extra_agent
    probabilities = probabilities.float()
    fire_decrease_probabilities[suppressant_needs_met] = probabilities[suppressant_needs_met]

    return torch.clamp(fire_decrease_probabilities, 0, 1)
