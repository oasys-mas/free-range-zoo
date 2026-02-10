"""Wildfire Environment v0 Transitions.

This module provides the original static wildfire environment transition classes.

Classes:
    CapacityTransition: Manages firefighter capacity constraints.
    EquipmentTransition: Handles firefighter equipment usage.
    FireDecreaseTransition: Reduces fire intensity based on suppression.
    FireIncreaseTransition: Increases fire intensity naturally.
    FireSpreadTransition: Models fire spread between cells.
    SuppressantDecreaseTransition: Reduces suppressant quantities.
    SuppressantRefillTransition: Refills suppressant supplies.
"""

from .transitions import (
    CapacityTransition,
    EquipmentTransition,
    FireDecreaseTransition,
    FireIncreaseTransition,
    FireSpreadTransition,
    SuppressantDecreaseTransition,
    SuppressantRefillTransition,
)

__all__ = [
    "CapacityTransition",
    "EquipmentTransition",
    "FireDecreaseTransition",
    "FireIncreaseTransition",
    "FireSpreadTransition",
    "SuppressantDecreaseTransition",
    "SuppressantRefillTransition",
]
