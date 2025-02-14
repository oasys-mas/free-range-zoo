"""Transition function for agent presence."""
import torch
from torch import nn

from free_range_zoo.envs.rideshare.env.structures.state import RideshareState


class PassengerStateTransition(nn.Module):
    """
    """

    def __init__(self, num_agents: int, parallel_envs: int) -> None:
        """
        Initialize the transition function.

        Args:
        """
        super().__init__()

        self.register_buffer('env_range', torch.arange(0, parallel_envs, dtype=torch.int32))
        self.register_buffer('agent_range', torch.arange(0, num_agents, dtype=torch.int32))

    @torch.no_grad()
    def forward(
        self,
        state: RideshareState,
        accepts: torch.BoolTensor,
        picks: torch.BoolTensor,
        targets: torch.IntTensor,
        vectors: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> RideshareState:
        """
        Calculate the next presence states for all agents.

        Args:
            state: RideshareState - the current state of the environment
            accepts: torch.BoolTensor - a mask over agents which dictates which agents have taken the accept action
            picks: torch.BoolTensor - a mask over agents which dictates which agents have taken the pick action
            targets: torch.IntTensor - the task targets of all agents for all environments
            vectors: torch.IntTensor - the point vectors for each agent movement in the form of (y, x, y_dest, x_dest)
            timesteps: torch.IntTensor - the timestep of each of the parallel environments
        Returns:
            RideshareState - the next state of the environment with the presence states transformed
        """
        # Aggregate vectors to determine distance between the subject and target
        distances = ((vectors[:, :, [0, 1]] - vectors[:, :, [2, 3]])**2).sum(dim=-1).sqrt()
        distances = torch.where((vectors == -100).all(dim=-1), torch.inf, distances)

        accept_targets = torch.where(accepts, targets, -100)

        # PERF: DO NOT TOUCH THIS IT IS THE OPTIMAL METHOD OF DOING THIS
        # Systematically remove duplicates iteratively and distribute accepts to the closest driver
        while True:
            accept_targets_flat = accept_targets.flatten()
            unique_vals, inverse_indices, counts = accept_targets_flat.unique(
                sorted=False,
                return_inverse=True,
                return_counts=True,
            )

            is_duplicated = (counts[inverse_indices].reshape(accept_targets.shape) > 1) & (accept_targets != -100)

            if not is_duplicated.any():
                break

            distances = torch.where(is_duplicated, distances, torch.inf)

            row_indices = torch.arange(accept_targets.shape[0], device=accept_targets.device).unsqueeze(1)
            keep_mask = torch.zeros_like(accept_targets, dtype=torch.bool)
            keep_mask[row_indices, torch.argmin(distances, dim=1, keepdim=True)] = True

            replace_mask = is_duplicated & ~keep_mask
            accept_targets[replace_mask] = -100

        # Set accepted passengers to the accepted state and timesteps
        indices = accept_targets[accept_targets != -100]
        state.passengers[:, 6][indices] = 1
        state.passengers[:, 9][indices] = timesteps[state.passengers[:, 0][indices]]

        # Update the associations of passengers to their respective agent
        agent_indices = self.agent_range[None, :].expand_as(accept_targets)
        agent_indices = torch.where(accept_targets != -100, agent_indices, -100)
        state.passengers[:, 7][indices] = agent_indices[agent_indices != -100]

        # Aggregate vectors to determine distance between the subject and target
        distances = ((vectors[:, :, [0, 1]] - vectors[:, :, [2, 3]])**2).sum(dim=-1).sqrt()
        distances = torch.where((vectors == -100).all(dim=-1), torch.inf, distances)

        # Determine the pick targets which are already at their destination
        pick_targets = torch.where(picks, targets, -100)
        pick_distances = torch.where(pick_targets != -100, distances, torch.inf)
        pick_targets = torch.where(pick_distances < 1e-6, pick_targets, -100)

        # Update the association of those agents
        indices = pick_targets[pick_targets != -100]
        state.passengers[:, 6][indices] = 2
        state.passengers[:, 10][indices] = timesteps[state.passengers[:, 0][indices]]

        return state
