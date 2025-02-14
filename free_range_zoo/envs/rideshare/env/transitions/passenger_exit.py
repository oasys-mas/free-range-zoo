"""Transition function for agent presence."""
import torch
from torch import nn

from free_range_zoo.envs.rideshare.env.structures.state import RideshareState


class PassengerExitTransition(nn.Module):
    """Transition for passenger exit and movement."""

    def __init__(self, parallel_envs: int) -> None:
        """
        Initialize the transition function.

        Args:
            parallel_envs: int - Number of parallel environments in the simulation.
        """
        super().__init__()

        self.register_buffer('env_range', torch.arange(0, parallel_envs, dtype=torch.int32))

    @torch.no_grad()
    def forward(self, state: RideshareState, drops: torch.BoolTensor, targets: torch.IntTensor, vectors: torch.FloatTensor,
                timesteps: torch.IntTensor) -> RideshareState:
        """
        Calculate the next presence states for all agents.

        Args:
            state: RideshareState - the current state of the environment
            drops: torch.BoolTensor - a mask over agents which dictates which agents have taken the drop action
            targets: torch.IntTensor - the task targets of all agents for all environments
            vectors: torch.IntTensor - the point vectors for each agent movement in the form of (y, x, y_dest, x_dest)
            timesteps: torch.IntTensor - the timestep of each of the parallel environments
        Returns:
            RideshareState - the next state of the environment with the presence states transformed
        """
        # Update targets to only include tasks which are being dropped
        targets = torch.where(drops, targets, -100)

        # Determine which tasks are at their target location
        distances = ((vectors[:, :, [0, 1]] - vectors[:, :, [2, 3]])**2).sum(dim=-1).sqrt()
        distances = torch.where((vectors == -100).all(dim=-1), torch.inf, distances) == 0

        # Update targets to only include tasks which are at their final destination
        targets = torch.where(drops & distances, targets, -100)

        # Distribute rewards for fares to each of the agents
        rewards = torch.zeros_like(targets, dtype=torch.int32, device=targets.device)
        rewards[targets != -100] = state.passengers[:, 5][targets[targets != -100]]

        # Mask out the completed tasks
        finished_mask = torch.ones((state.passengers.size(0), ), dtype=torch.bool, device=targets.device)
        finished_mask[targets[targets != -100]] = 0
        state.passengers = state.passengers[finished_mask]

        return state, rewards
