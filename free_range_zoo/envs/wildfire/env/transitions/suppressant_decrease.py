import torch


def calculate_mask(targets: torch.Tensor,
                   stochastic_transition: bool,
                   decrease_probability: float,
                   randomness_source: torch.Tensor,
                   device: torch.DeviceObjType = torch.device('cpu')) -> torch.Tensor:
    """
    Transition function for stoachastic suppressant decrease

    Args:
        targets: torch.Tensor - Locations of agents that used suppressants
        stochastic_transition: bool - Whether to use stochastic transition
        decrease_probability: float - Decrease probability
        randomness_source: torch.Tensor - Randomness source
        device: torch.DeviceObjType - Device to move tensors to
    Returns:
        torch.Tensor - Mask for suppressant_decrease
    """
    indices = targets.split(1, dim=1)

    mask = torch.zeros_like(randomness_source, dtype=torch.bool, device=device)
    if not stochastic_transition:
        mask[indices] = True
        return mask

    mask[indices] = randomness_source[indices] < decrease_probability
    return mask
