"""Abstract state class for storing environmental state."""
from __future__ import annotations
from typing import Self, Optional, Tuple, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import copy
import torch
import pandas as pd


@dataclass
class State(ABC):
    """Abstract state class for storing environmental state."""

    def __post_init__(self):
        """Run after initialization actions."""
        self.metadata = {}
        self.initial_state = None
        self.checkpoint = None

    def to(self, device: torch.DeviceObjType = torch.device('cpu')) -> Self:
        """
        Move all tensors to the specified device.

        Args:
            device: torch.DeviceObjType - Device to move tensors to
        Returns:
            Self - The modified configuration
        """
        for attribute, value in self.__dict__.items():
            if hasattr(value, 'to'):
                setattr(self, attribute, value.to(device))

        return self

    def save_initial(self):
        """Save the initial state."""
        self.initial_state = self.clone()

    def restore_initial(self, batch_indices: Optional[torch.Tensor] = None) -> None:
        """
        Restore the initial state of the environment.

        Args:
            batch_indices: Optional[torch.Tensor] - The indices of the batch to restore
        """
        if self.initial is None:
            raise ValueError("Initial state is not saved")

        if batch_indices is None:
            self.__dict__ = self.initial_state.__dict__
        else:
            for attribute, value in self.initial_state.__dict__.items():
                if hasattr(value, 'clone'):
                    current_value = getattr(self, attribute)
                    current_value[batch_indices] = value[batch_indices]
                    setattr(self, attribute, current_value)
                else:
                    setattr(self, attribute, value)

    def save_checkpoint(self):
        """Save the current state as a checkpoint."""
        self.checkpoint = self.clone()

    def restore_from_checkpoint(self, batch_indices: Optional[torch.Tensor] = None) -> None:
        """
        Restore the state from the checkpoint.

        Args:
            batch_indices: Optional[torch.Tensor] - The indices of the batch to restore
        """
        if self.checkpoint is None:
            raise ValueError("Checkpoint is not saved")

        if batch_indices is None:
            self.__dict__ = self.checkpoint.__dict__
        else:
            for attribute, value in self.checkpoint.__dict__.items():
                if hasattr(value, 'clone'):
                    current_value = getattr(self, attribute)
                    current_value[batch_indices] = value[batch_indices]
                    setattr(self, attribute, current_value)
                else:
                    setattr(self, attribute, value)

    def load_state(self, state: Self, batch_indices: Optional[torch.Tensor] = None) -> None:
        """
        Load a custom state.

        Args:
            state: Self - The state to load
            batch_indices: Optional[torch.Tensor] - The indices of the batch to load
        """
        if batch_indices is None:
            self.__dict__ = state.__dict__
        else:
            for attribute, value in state.__dict__.items():
                if hasattr(value, 'clone'):
                    current_value = getattr(self, attribute)
                    for index, batch_index in enumerate(batch_indices):
                        current_value[batch_index] = value[index]
                    setattr(self, attribute, current_value)
                else:
                    setattr(self, attribute, value)

    def clone(self) -> Self:
        """
        Clone the state.

        Returns:
            Self - The cloned state
        """
        cloned_attributes = {}
        for attribute, value in self.__dict__.items():
            if hasattr(value, 'clone'):
                cloned_attributes[attribute] = value.clone()
            else:
                cloned_attributes[attribute] = copy.deepcopy(value)
        cloned_initial = cloned_attributes.pop('initial_state', None)
        cloned_checkpoint = cloned_attributes.pop('checkpoint', None)
        cloned_metadata = cloned_attributes.pop('metadata', None)

        cloned = self.__class__(**cloned_attributes)
        cloned.initial_state = cloned_initial
        cloned.checkpoint = cloned_checkpoint
        cloned.metadata = cloned_metadata

        return cloned

    @staticmethod
    def stack(states: List[State], *args, **kwargs) -> Self:
        """
        Stack a list of states.

        Args:
            states: List[State] - The states to stack
            args: Any - Additional arguments for torch.stack
            kwargs: Any - Additional keyword arguments for torch.stack
        Returns:
            Self - The stacked states
        """
        stacked_attributes = {}
        for attribute in states[0].__dict__.keys():
            if attribute in ('initial_state', 'checkpoint', 'metadata'):
                continue
            if attribute == 'agents':
                stacked_attributes[attribute] = getattr(states[0], attribute)
                continue
            stacked_attributes[attribute] = torch.stack([getattr(state, attribute) for state in states], *args, **kwargs)

        stacked = states[0].__class__(**stacked_attributes)

        return stacked

    @staticmethod
    def cat(states: List[State], *args, **kwargs) -> Self:
        """
        Concatenate a list of batched state tensors.

        Args:
            states: List[State] - The states to stack
            args: Any - Additional arguments for torch.stack
            kwargs: Any - Additional keyword arguments for torch.stack
        Returns:
            Self - The stacked states
        """
        concatenated = {}
        for attribute in states[0].__dict__.keys():
            if attribute in ('initial_state', 'checkpoint', 'metadata'):
                continue
            if attribute == 'agents':
                concatenated[attribute] = getattr(states[0], attribute)
                continue
            concatenated[attribute] = torch.cat([getattr(state, attribute) for state in states], *args, **kwargs)

        concat = states[0].__class__(**concatenated)

        return concat

    def to_dataframe(self) -> int:
        """Convert the state into a dataframe."""
        shared = self.metadata.get('shared', ())
        blacklist = ('initial_state', 'checkpoint', 'metadata') + shared
        columns = [attribute for attribute in self.__dict__.keys() if attribute not in blacklist]
        data = {column: [str(row.tolist()) for row in self.__dict__[column]] for column in columns}

        df = pd.DataFrame(data)
        for key in shared:
            df[key] = str(self.__dict__[key].tolist())

        return df

    def unwrap(self) -> List[Self]:
        """
        Unwrap a set of batched states.

        Returns:
            List[Self] - The unwrapped states
        """
        unwrapped_states = []

        for index in range(len(self)):
            unwrapped_state = self[index]
            unwrapped_states.append(unwrapped_state)

        return unwrapped_states

    def __len__(self) -> int:
        """
        Get the length of the state.

        Returns:
            int - The length of the state
        """
        for attribute in self.__dict__.values():
            if hasattr(attribute, '__len__'):
                return len(attribute)

    @abstractmethod
    def __getitem__(self, indices: torch.Tensor) -> Self:
        """
        Get the state at the specified indices.

        Args:
            indices: torch.Tensor - The indices to get
        Returns:
            Self - The state at the specified indices
        """
        raise NotImplementedError("State must implement __getitem__ method.")

    @abstractmethod
    def __hash__(self) -> Tuple[int] | int:
        """
        Hash the state.

        Returns:
            Tuple[int] | int - The hash of the state
        """
        raise NotImplementedError("State must implement __hash__ method.")
