from typing import Tuple
import torch


def calculate_modified(targets: torch.Tensor,
                       stochastic_transition: bool,
                       tank_switch_probability: float,
                       possible_capacities: torch.Tensor,
                       capacity_probabilities: torch.Tensor,
                       randomness_source: torch.Tensor,
                       device: torch.DeviceObjType = torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the possible new modified maximum suppressant capacities

    Args:
        targets: torch.Tensor - Indices of agents which have successfully refilled their suppressants
        stochastic_transition: bool - Whether to switch the maximum suppressant value
        tank_switch_probability: float - Probability of switching the maximum suppressant value
        possible_suppressant_maximums: torch.Tensor - Possible maximum suppressant values
        suppressant_maximum_probabilities: torch.Tensor - Probabilities of each maximum suppressant value
        randomness_source: torch.Tensor - Randomness source
    Returns:
        torch.Tensor - Mask for which agents should recieve a new maximum suppressant capacity
        torch.Tensor - New maximum suppressant capacities for all agents
    """
    indices = targets.split(1, dim=1)

    cum_capacity_probability = torch.cumsum(capacity_probabilities, dim=0)

    size_indices = torch.bucketize(randomness_source[:, 0], cum_capacity_probability)
    new_maximums = possible_capacities[size_indices]

    tank_switch_mask = torch.zeros_like(randomness_source[:, 1], dtype=torch.bool, device=device)
    if not stochastic_transition:
        tank_switch_mask[indices] = True
        return tank_switch_mask, new_maximums

    tank_switch_mask[indices] = randomness_source[:, 1][indices] < tank_switch_probability[indices]
    return tank_switch_mask, new_maximums
