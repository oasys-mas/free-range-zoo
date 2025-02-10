"""Transition function for agent presence."""
import torch
from torch import nn

from free_range_zoo.envs.rideshare.env.structures.state import RideshareState


class PassengerEntryTransition(nn.Module):
    """
    """

    def __init__(self, schedule: torch.IntTensor, parallel_envs: int) -> None:
        """
        Initialize the transition function.

        Args:
        """
        super().__init__()

        self.register_buffer('schedule', schedule)
        self.register_buffer('env_range', torch.arange(0, parallel_envs, dtype=torch.int32))

    @torch.no_grad()
    def forward(self, state: RideshareState, timesteps: torch.IntTensor) -> RideshareState:
        """
        Calculate the next presence states for all agents.

        Args:
            state: RideshareState - the current state of the environment
            timesteps: torch.IntTensor - the timestep of each of the parallel environments
        Returns:
            RideshareState - the next state of the environment with the presence states transformed
        """
        # Determine which tasks within the environment enter this step
        source = self.schedule[:, :2]
        target = torch.cat([timesteps.unsqueeze(1), self.env_range.unsqueeze(1)], dim=1)

        source_expanded = source[:, None, :]
        target_expanded = target[None, :, :]
        first_column_match = source_expanded[:, :, 0] == target_expanded[:, :, 0]
        second_column_match = (source_expanded[:, :, 1] == target_expanded[:, :, 1]) | (source_expanded[:, :, 1] == -1)
        schedule_mask = (first_column_match & second_column_match).any(dim=1)

        entered_tasks = self.schedule[schedule_mask]

        # Repeat the entered tasks per-environment if necessary
        if entered_tasks.size(0) > 0:
            repeat = entered_tasks[:, 1] == -1
            repeated_rows = entered_tasks[repeat].repeat_interleave(self.env_range.size(0), dim=0)
            repeated_rows[:, 1] = self.env_range

            remaining_rows = entered_tasks[~repeat]

            # Build the task entries to add to the task store
            entered = torch.cat([remaining_rows, repeated_rows], dim=0)[:, 1:]
            new_cols = torch.empty((entered.size(0), 5), device=entered.device, dtype=torch.int32)

            task_store = torch.cat([entered, new_cols], dim=1)

            # Set the state of each task to unaccepted
            task_store[:, 6] = 0

            # Set each task to unassociated with any driver
            task_store[:, 7] = -1

            # Set the timestep that each task entered the environment
            task_store[:, 8] = timesteps[task_store[:, 0]]

            # Set the timestep that each task was accept and picked to null for now
            task_store[:, 9] = -1
            task_store[:, 10] = -1

            # Add the passengers just queried into the state
            if state.passengers is None:
                state.passengers = task_store
            else:
                state.passengers = torch.cat([state.passengers, task_store], dim=0)

            state.passengers = state.passengers[torch.argsort(state.passengers[:, 0])]

        return state
