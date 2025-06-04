"""Passthrough modifier fallback class so that modifiers can be selectively applied to specific agents."""
from supersuit.generic_wrappers.utils.base_modifier import BaseModifier

from free_range_zoo.utils.env import BatchedAECEnv


class PassthroughWrapperModifier(BaseModifier):
    """Passthrough modifier fallback class so that modifiers can be selectively applied to specific agents."""

    def __init__(self):
        """
        Initialize the ActionTaskMappingWrapperModifier.
        """
        pass
