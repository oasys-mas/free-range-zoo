"""Transition function for agent presence."""
import torch
from torch import nn

from free_range_zoo.envs.rideshare.env.structures.state import RideshareState


class MovementTransition(nn.Module):
    """
    """

    def __init__(self, parallel_envs: int, num_agents: int, fast_travel: bool, diagonal_travel: bool) -> None:
        """
        Initialize the transition function.

        Args:
        """
        super().__init__()

        self.fast_travel = fast_travel

        self.register_buffer('env_range', torch.arange(0, parallel_envs, dtype=torch.int32))
        self.register_buffer('agent_range', torch.arange(0, num_agents, dtype=torch.int32))

        # Directions for 4-connected and 8-connected grid movement
        cardinal_directions = torch.tensor(
            [
                [0, 0],  # No movement
                [-1, 0],  # N
                [0, 1],  # E
                [1, 0],  # S
                [0, -1]  # W
            ],
            dtype=torch.int32,
        )

        diagonal_directions = torch.tensor(
            [
                [-1, -1],  # NW
                [-1, 1],  # NE
                [1, 1],  # SE
                [1, -1],  # SW
            ],
            dtype=torch.int32,
        )

        if diagonal_travel:
            self.register_buffer('directions', torch.cat([cardinal_directions, diagonal_directions], dim=0))
        else:
            self.register_buffer('directions', cardinal_directions)

    @torch.no_grad()
    def forward(self, state: RideshareState, mask: torch.BoolTensor, vectors: torch.IntTensor) -> RideshareState:
        """
        Calculate the next presence states for all agents.

        Args:
            state: RideshareState - the current state of the environment
            timesteps: torch.IntTensor - the timestep of each of the parallel environments
        Returns:
            RideshareState - the next state of the environment with the presence states transformed
        """
        current_positions = vectors[:, :, :2]
        target_positions = vectors[:, :, 2:]

        candidate_positions = current_positions.unsqueeze(2) + self.directions.view(1, 1, -1, 2)
        distances = torch.norm((candidate_positions - target_positions.unsqueeze(2)).float(), dim=3)

        best_moves = torch.argmin(distances, dim=2, keepdim=True)
        best_moves = self.directions[best_moves.squeeze(-1)]
        best_moves[current_positions == -100] = 0

        state.agents += best_moves

        passenger_indices = state.passengers[:, [0, 7]].T.split(1, dim=0)
        passenger_movements = best_moves[passenger_indices].squeeze(0)
        passenger_movements[state.passengers[:, 7] == -1][:, 0] = 0
        state.passengers[:, 1:3] += passenger_movements

        distances = best_moves.float().norm(dim=2)
        return state, distances
