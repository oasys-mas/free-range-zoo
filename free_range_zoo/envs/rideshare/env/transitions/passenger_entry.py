"""Transition function for agent presence."""
import torch
from torch import nn

from free_range_zoo.envs.rideshare.env.structures.state import RideshareState


class PassengerEntryTransition(nn.Module):
    """Transition for adding tasks to the environment according to schedule."""

    def __init__(self, schedule: torch.IntTensor, parallel_envs: int) -> None:
        """
        Initialize the transition function.

        Args:
            parallel_envs: int - the number of parallel environments to vectorize operations over
            schedule: torch.IntTensor - schedule from the environment configuration to determine task entry
        """
        super().__init__()

        self.register_buffer('schedule', schedule)
        self.register_buffer('env_range', torch.arange(0, parallel_envs, dtype=torch.int32))

    @torch.no_grad()
    def forward(self, state: RideshareState, timesteps: torch.IntTensor) -> RideshareState:
        """
        Calculate the next passengers to enter each environment using the schedule.

        Args:
            state: RideshareState - the current state of the environment
            timesteps: torch.IntTensor - the timestep of each of the parallel environments
        Returns:
            RideshareState - the next state of the environment with new passengers added
        """
        # Determine which tasks within the environment enter this step
        task_start_times = self.schedule[:, 0]
        task_env_ids = self.schedule[:, 1]

        timesteps_expanded = timesteps.unsqueeze(0)
        env_range_expanded = self.env_range.unsqueeze(0)

        starts_now = task_start_times[:, None] == timesteps_expanded
        valid_env = (task_env_ids[:, None] == env_range_expanded) | (task_env_ids[:, None] == -1)

        entry_mask = starts_now & valid_env
        if not entry_mask.any():
            if state.passengers is None:
                state.passengers = torch.empty((0, 11), dtype=torch.int32, device=self.schedule.device)

            return state

        task_indices, batch_indices = torch.where(entry_mask)

        entered_tasks = self.schedule[task_indices].clone()
        entered_tasks[:, 1] = self.env_range[batch_indices]

        num_new_tasks = entered_tasks.size(0)
        new_task_data = torch.full((num_new_tasks, 5), -1, device=entered_tasks.device, dtype=torch.int32)

        task_store = torch.cat([entered_tasks[:, 1:], new_task_data], dim=1)

        # Set remaining task attributes
        task_store[:, 6] = 0  # Task state: unaccepted
        task_store[:, 7] = -1  # Driver association: None
        task_store[:, 8] = timesteps[task_store[:, 0]]  # Task entry timestep
        task_store[:, 9] = -1  # Task accepted timestep
        task_store[:, 10] = -1  # Task picked timestep

        state.passengers = task_store if state.passengers is None else torch.cat([state.passengers, task_store], dim=0)
        state.passengers = state.passengers[torch.argsort(state.passengers[:, 0], stable=True)]

        return state
