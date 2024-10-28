from __future__ import annotations

from typing import Self, Optional, Tuple, List, Union, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import copy, os

import pandas as pd


@dataclass
class State(ABC):

    def __post_init__(self):
        self.initial_state = None
        self.checkpoint = None

    def log(self,
            path: str,
            new_episode: bool = False,
            constant_observations: Optional[List[str]] = [],
            initial: Optional[bool] = False,
            label: Optional[str] = None,
            partial_log: Optional[List[int]] = None,
            actions: Union[Dict[str, torch.Tensor], List[str]] = None,
            rewards: Union[Dict[str, torch.Tensor], List[float]] = None,
            infos: Union[Dict[str, torch.Tensor], List[str]] = None,
            log_exclusions: List[str] = [],
            masked_attributes: Dict[Tuple[str, int], torch.Tensor] = None):
        """
        Save the state to log files

        Only includes constants on the first line, then the empty on the rest

        Args:
            path: str - Path to save the log files (will split log files into seperate based on batches)
            new_episode: bool - is this a new episode?
            constant_observations: Optional[List[str] - list of attributes that are constant throughout the episode
            initial: bool - use file headers
            label: Optional[str] - a generic string filled in as the "label" column
            partial_log: Optional[List[int]] - list of specific batch indices to log
            actions: Union[Dict[str, torch.Tensor], List[str]] - dictionary of actions to log
            exclusions: List[str] - list of attributes to exclude from logging
            masked_attributes: Dict[Tuple[str, int], torch.Tensor] - dictionary of attributes (per batch) to mask rather than batch index
        """

        if initial:
            try:
                os.mkdir(path)
            except FileExistsError:
                assert not os.path.exists(os.path.join(
                    path, "0.csv")), "path already exists and files found, check path. Don't waste experiments!"

        if label is not None:
            assert "test" not in label and "train" not in label, "label should not be used to distinguish between test and train data make a new file"

        #ensure all elements are tensors (in case of nested states)
        random_variables = {
            key: value
            for key, value in self.__dict__.items()
            if isinstance(value, torch.Tensor) and key not in constant_observations and key not in log_exclusions
        }

        #all elements have the same batch size
        batch_size = random_variables[list(random_variables.keys())[0]].shape[0]
        assert all([random_variables[key].shape[0] == random_variables[list(random_variables.keys())[0]].shape[0]\
             for key in random_variables.keys()]), "All elements must have the same batch size, check constant_observations list"

        #?handle defaults / env init states
        if isinstance(actions, dict):
            actions = {key: value.tolist() for key, value in actions.items()}
        else:
            actions = {key: [None for _ in range(batch_size)] for key in actions}
        if isinstance(rewards, dict):
            rewards = {key: value.tolist() for key, value in rewards.items()}
        else:
            rewards = {key: [0.0 for _ in range(batch_size)] for key in rewards}
        
        infos = {key: value if value!={} else {'task-action-indices':[None for _ in range(batch_size)]} for key, value in infos.items()}
        

        if partial_log is None:
            random_variables = {key: value.tolist() for key, value in random_variables.items()}

        elif (isinstance(partial_log, bool) and partial_log):
            random_variables = {key: value.tolist() for key, value in random_variables.items()}
            partial_log = None

        else:
            random_variables = {key: value[partial_log].tolist() for key, value in random_variables.items()}

        #?handling initial observations (present at all times, but storing sparsely)
        if initial:
            constants = {
                key: value
                for key, value in self.__dict__.items() if isinstance(value, torch.Tensor) and key in constant_observations
            }
            constants = {key: [value.tolist()] for key, value in constants.items()}
        else:
            constants = {key: None for key in constant_observations}

        if partial_log is None:
            batch_loop_list = list(range(batch_size))
        else:
            batch_loop_list = partial_log

        #?saving to log files per batch
        for batch in batch_loop_list:

            batch_random_variables = {key: [value[batch]] for key, value in random_variables.items()}

            batched_actions = {k: [v[batch]] for k, v in actions.items()}
            batched_rewards = {k + "_rewards": [v[batch]] for k, v in rewards.items()}

            batched_info = {}
            for _ag, _ag_infos in infos.items():
                for _info_key, _info_value in _ag_infos.items():
                    batched_info[f"{_ag}_{_info_key}"] = [v[batch] for v in _info_value]

            data = batch_random_variables | constants | batched_actions | batched_rewards | batched_info

            df = pd.DataFrame(data)

            df['label'] = label
            df['new_episode'] = new_episode

            df.to_csv(os.path.join(path, f"{batch}.csv"), mode='a' if not initial else 'w', header=initial, index=False)

    def to(self, device: torch.DeviceObjType = torch.device('cpu')) -> Self:
        """
        Move all tensors to the specified device
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
        """
        Save the initial state
        """
        self.initial_state = self._clone()

    def restore_initial(self, batch_indices: Optional[torch.Tensor] = None) -> None:
        """
        Restore the initial state of the environment

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
        """
        Save the current state as a checkpoint
        """
        self.checkpoint = self._clone()

    def restore_from_checkpoint(self, batch_indices: Optional[torch.Tensor] = None) -> None:
        """
        Restore the state from the checkpoint

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
        Load a custom state

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

    def _clone(self) -> Self:
        """
        Clone the state

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

        cloned = self.__class__(**cloned_attributes)
        cloned.initial_state = cloned_initial
        cloned.checkpoint = cloned_checkpoint

        return cloned

    @staticmethod
    def stack(states: List[State], *args, **kwargs) -> Self:
        """
        Stack a list of states

        Args:
            states: List[State] - The states to stack
            args: Any - Additional arguments for torch.stack
            kwargs: Any - Additional keyword arguments for torch.stack
        Returns:
            Self - The stacked states
        """
        stacked_attributes = {}
        for attribute in states[0].__dict__.keys():
            if attribute in ['initial_state', 'checkpoint']:
                continue
            if attribute in ['agents']:
                stacked_attributes[attribute] = getattr(states[0], attribute)
                continue
            stacked_attributes[attribute] = torch.stack([getattr(state, attribute) for state in states], *args, **kwargs)

        stacked = states[0].__class__(**stacked_attributes)

        return stacked

    def unwrap(self) -> List[Self]:
        """
        Unwrap a set of batched states
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
        Get the length of the state
        Returns:
            int - The length of the state
        """
        for attribute in self.__dict__.values():
            if hasattr(attribute, '__len__'):
                return len(attribute)

    @abstractmethod
    def __getitem__(self, indices: torch.Tensor) -> Self:
        """
        Get the state at the specified indices

        Args:
            indices: torch.Tensor - The indices to get
        Returns:
            Self - The state at the specified indices
        """
        raise NotImplementedError("State must implement __getitem__ method.")

    @abstractmethod
    def __hash__(self) -> Tuple[int] | int:
        """
        Hash the state

        Returns:
            Tuple[int] | int - The hash of the state
        """
        raise NotImplementedError("State must implement __hash__ method.")
