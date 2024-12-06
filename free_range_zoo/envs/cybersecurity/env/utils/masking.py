"""Utilities for masking observations for both defenders and attackers."""
from functools import lru_cache
import torch


@lru_cache(maxsize=100)
def mask_observation(agent_name: str, observe_other_power: bool, observe_other_presence: bool, observe_other_location: bool):
    """
    Mask the observation for the agent.

    Args:
        agent_name: str - Name of the agent
        observe_other_power: bool - Whether to observe the power of other agents
        observe_other_presence: bool - Whether to observe the presence of other agents
        observe_other_location: bool - Whether to observe the location of other agents
    """
    match agent_name.split('_')[0]:
        case 'defender':
            defender_observation_mask = torch.ones(3, dtype=torch.bool)
            defender_observation_mask[0] = observe_other_power
            defender_observation_mask[1] = observe_other_presence
            defender_observation_mask[2] = observe_other_location
            return defender_observation_mask
        case 'attacker':
            attacker_observation_mask = torch.ones(2, dtype=torch.bool)
            attacker_observation_mask[0] = observe_other_power
            attacker_observation_mask[1] = observe_other_presence
            return attacker_observation_mask
