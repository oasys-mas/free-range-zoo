"""Transition function for agent presence."""
from typing import Tuple
import torch
from torch import nn

from free_range_zoo.envs.rideshare.env.structures.state import RideshareState


class MovementTransition(nn.Module):
    """Movement transition for agents and tasks within the rideshare environment."""

    def __init__(self, parallel_envs: int, num_agents: int, fast_travel: bool, diagonal_travel: bool) -> None:
        """
        Initialize the transition function.

        Args:
            parallel_envs: int - the number of parallel environments to vectorize operations over
            num_agents: int - the number of agents to expect in each environment
            fast_travel: bool - whether to enable fast travel for agents
            diagonal_travel: bool - whether to enable diagonal 8 direction movement for agents
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
    def distance(self, starts: torch.IntTensor, goals: torch.IntTensor) -> Tuple[torch.IntTensor, torch.FloatTensor]:
        """
        Calculate the distance between two sets of positions.

        Args:
            starts: torch.IntTensor - The starting positions of the agents
            goals: torch.IntTensor - The goal positions of the agents
        Returns:
            best_moves: torch.IntTensor - The best moves for each agent
            distances: torch.IntTensor - The distance traveled for each agent
        """
        candidate_positions = starts.unsqueeze(2) + self.directions.view(1, 1, -1, 2)
        distances = torch.norm((candidate_positions - goals.unsqueeze(2)).float(), dim=3)

        if self.fast_travel:
            best_moves = starts - goals
        else:
            best_moves = torch.argmin(distances, dim=2, keepdim=True)
            best_moves = self.directions[best_moves.squeeze(-1)]
        best_moves[starts == -100] = 0

        # Calculate the movement distances so that these can be used to calculate costs later
        distances = best_moves.float()
        match self.directions.size(0):
            case 9:
                distances = distances.norm(dim=2)
            case 5:
                distances = distances.abs().sum(dim=2)

        return best_moves, distances

    @torch.no_grad()
    def forward(self, state: RideshareState, vectors: torch.IntTensor) -> RideshareState:
        """
        Calculate the next presence states for all agents.

        Args:
            state: RideshareState - the current state of the environment
            vectors: torch.IntTensor - the point vectors for each agent in the form of (y, x, y_dest, x_dest)
        Returns:
            RideshareState - the next state of the environment with the modified movement positions
        """
        current_positions = vectors[:, :, :2]
        target_positions = vectors[:, :, 2:]

        # Figure out where agents should be moving
        best_moves, distances = self.distance(current_positions, target_positions)
        state.agents += best_moves

        # Move passengers along with their agents
        passenger_indices = state.passengers[:, [0, 7]].unbind(1)
        passenger_movements = best_moves[passenger_indices].squeeze(1)

        # Figure out which passengers should not be moving
        no_movement = (state.passengers[:, 6] == 0) | (state.passengers[:, 6] == 1)
        passenger_movements[no_movement] = 0

        state.passengers[:, 1:3] += passenger_movements

        return state, distances
