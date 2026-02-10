"""Factory module for creating cybersecurity v0 transitions.

This module provides functions to create configured transition instances for the cybersecurity environment.
"""
from typing import Tuple

import torch

from free_range_zoo.envs.cybersecurity.env.v0.structures.configuration import CybersecurityConfiguration
from free_range_zoo.envs.cybersecurity.env.v0.transitions.movement import MovementTransition
from free_range_zoo.envs.cybersecurity.env.v0.transitions.presence import PresenceTransition
from free_range_zoo.envs.cybersecurity.env.v0.transitions.subnetwork import SubnetworkTransition


def create_cybersecurity_transitions(
    config: CybersecurityConfiguration,
    device: torch.DeviceObjType = None,
) -> Tuple[MovementTransition, PresenceTransition, SubnetworkTransition]:
    """
    Create all configured transitions for the cybersecurity environment.

    Args:
        config (CybersecurityConfiguration): Cybersecurity configuration.
        device (torch.DeviceObjType): Device to place transitions on.

    Returns:
        Tuple: Tuple of (movement, presence, subnetwork) transitions on the specified device.
    """
    movement = create_movement_transition(config)
    presence = create_presence_transition(config)
    subnetwork = create_subnetwork_transition(config)

    if device is not None:
        movement = movement.to(device)
        presence = presence.to(device)
        subnetwork = subnetwork.to(device)

    return movement, presence, subnetwork


def create_movement_transition(config: CybersecurityConfiguration) -> MovementTransition:
    """Create a configured movement transition.

    Args:
        config (CybersecurityConfiguration): Cybersecurity configuration.

    Returns:
        MovementTransition: Configured MovementTransition.
    """
    return MovementTransition()


def create_presence_transition(config: CybersecurityConfiguration) -> PresenceTransition:
    """Create a configured presence transition.

    Args:
        config (CybersecurityConfiguration): Cybersecurity configuration.

    Returns:
        PresenceTransition: Configured PresenceTransition.
    """
    return PresenceTransition(
        persist_probs=config.persist_probs,
        return_probs=config.return_probs,
        num_attackers=config.attacker_config.num_attackers,
    )


def create_subnetwork_transition(config: CybersecurityConfiguration) -> SubnetworkTransition:
    """Create a configured subnetwork transition.

    Args:
        config (CybersecurityConfiguration): Cybersecurity configuration.

    Returns:
        SubnetworkTransition: Configured SubnetworkTransition.
    """
    return SubnetworkTransition(
        patched_states=config.network_config.patched_states,
        vulnerable_states=config.network_config.vulnerable_states,
        exploited_states=config.network_config.exploited_states,
        temperature=config.network_config.temperature,
        stochastic_state=config.stochastic_config.network_state,
    )


def create_all_transitions(
        config: "CybersecurityConfiguration") -> Tuple[MovementTransition, PresenceTransition, SubnetworkTransition]:
    """Create all configured transitions for the cybersecurity environment.

    Args:
        config (CybersecurityConfiguration): Cybersecurity configuration.

    Returns:
        Tuple: Tuple of (movement, presence, subnetwork) transitions.
    """
    return (
        create_movement_transition(config),
        create_presence_transition(config),
        create_subnetwork_transition(config),
    )


__all__ = [
    "create_movement_transition",
    "create_presence_transition",
    "create_subnetwork_transition",
    "create_all_transitions",
    "create_cybersecurity_transitions",
]
