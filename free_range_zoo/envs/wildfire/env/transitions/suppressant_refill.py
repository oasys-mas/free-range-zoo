import torch


def calculate_mask(targets: torch.Tensor,
                   stochastic_transition: bool,
                   refill_probability: float,
                   randomness_source: torch.Tensor,
                   device: torch.DeviceObjType = torch.device('cpu')) -> torch.Tensor:
    """
    Transition function for stochastic suppressant refills

    Args:
        targets: torch.Tensor - Refill locations
        stochastic_transition: bool - Whether to use stochastic transition
        refill_probability: float - Refill probability
        randomness_source: torch.Tensor - Randomness source
        device: torch.DeviceObjType - Device to move tensors to
    Returns:
        torch.Tensor - Updated suppressants tensor
    """
    indices = targets.split(1, dim=1)

    mask = torch.zeros_like(randomness_source, dtype=torch.bool, device=device)
    if not stochastic_transition:
        mask[indices] = True
        return mask

    mask[indices] = randomness_source[indices] < refill_probability
    return mask
