"""Transition function for agent position / movement."""
import torch
from torch import nn

from free_range_zoo.envs.cybersecurity.env.structures.state import CybersecurityState


class MovementTransition(nn.Module):
    """
    Transition function for agent position / movement.

    Agents can move to any nodes that are connected to the one that they are currently in.
    For efficiency, no validity checks are performed on the movement targets.
    """

    @torch.no_grad()
    def forward(self, state: CybersecurityState, movement_targets: torch.IntTensor,
                movement_mask: torch.BoolTensor) -> CybersecurityState:
        """
        Calculate the next presence states for all agents.

        Args:
            state: CybersecurityState - the current state of the environment
            movement_targets: CybersecurityState - the target location for agents that are moving
            movement_mask: torch.BoolTensor - the mask for agents that are moving
        Returns:
            CybersecurityState - the next state of the environment with the presence states transformed
        """
        # Update the locations of the agents that moved to their target location
        state.location[movement_mask] = movement_targets[movement_mask]

        return state
