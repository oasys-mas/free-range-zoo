import torch


def calculate_modified(equipment_conditions: torch.Tensor,
                       num_equipment_states: int,
                       repair_probability: float,
                       degrade_probability: float,
                       critical_error_probability: float,
                       stochastic_repair: bool,
                       stochastic_degrade: bool,
                       critical_error: bool,
                       randomness_source: torch.Tensor) -> torch.Tensor:
    """
    Calculate the new modified equipment conditions
        - if the equipment is pristine, it can degrade or have a critical error
        - if the equipment is damaged, it can be repaired
        - if the equipment is in an intermediate state, it can degrade

    Args:
        equipment_conditions: torch.Tensor - The equipment conditions
        num_equipment_states: int - The number of equipment states
        repair_probability: float - The probability of repair
        degrade_probability: float - The probability of degradation
        critical_error_probability: float - The probability of a critical error
        stochastic_repair: bool - Whether to have a stochastic repair
        stochastic_degrade: bool - Whether to have a stochastic degrade
        critical_error: bool - Whether to have a critical error
        randomness_source: torch.Tensor - Randomness source
    Returns:
        torch.Tensor - The new equipment conditions
    """
    pristine = equipment_conditions == (num_equipment_states - 1)
    damaged = equipment_conditions == 0
    intermediate = torch.logical_and(torch.logical_not(pristine), torch.logical_not(damaged))

    if not stochastic_repair:
        repairs = damaged
    else:
        repairs = torch.logical_and(damaged, randomness_source < repair_probability)

    equipment_conditions[repairs] = num_equipment_states - 1

    if critical_error:
        criticals = torch.logical_and(pristine, randomness_source < critical_error_probability)
        equipment_conditions[criticals] = 0

    if not stochastic_degrade:
        degrades = torch.logical_or(intermediate, pristine)
    else:
        degrades = torch.logical_and(torch.logical_or(pristine, intermediate), randomness_source < degrade_probability)

        if critical_error:
            degrades = torch.logical_and(degrades, torch.logical_not(criticals))

    equipment_conditions[degrades] -= 1

    return equipment_conditions
