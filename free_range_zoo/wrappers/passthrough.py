"""Passthrough Modifier for Selective Agent Application.

This module provides the PassthroughWrapperModifier class as a fallback
for selectively applying modifiers to specific agents without modification.
"""
from supersuit.generic_wrappers.utils.base_modifier import BaseModifier


class PassthroughWrapperModifier(BaseModifier):
    """Passthrough modifier fallback class so that modifiers can be selectively applied to specific agents.

    This modifier does nothing - it passes observations and actions through unchanged.
    Used as a fallback when a wrapper should not apply to certain agents.
    """

    def __init__(self):
        """Initialize the passthrough modifier."""
        pass
