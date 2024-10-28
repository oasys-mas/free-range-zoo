import os
from typing import Self, Tuple, Union, Optional, List, Dict, Any
from dataclasses import dataclass

import pandas as pd

import torch

from free_range_zoo.utils.state import State
from free_range_zoo.utils.caching import optimized_convert_hashable


@dataclass
class RideshareState(State):
    """
    
    agents: torch.Tensor - Tensor representing the location of each agent <Z, 4>    batch_index, agent_indx, y, x
    locations: torch.Tensor - Tensor representing the location of each passenger <Z, 3> batch_index, y, x
    destinations: torch.Tensor - Tensor representing the location of each destination <Z, 3>  batch_index, dest_y, dest_x
    associations: torch.Tensor - Tensor representing the location of each passenger <Z, 4>   batch_index, accepted_by, riding_with, fare
    timing: torch.IntTensor - Tensor representing the time each state transition occured <Z, 5> batch_index, entered_at, accepted, picked_up, dropped_off
    """
    agents: torch.IntTensor

    locations: torch.IntTensor
    destinations: torch.IntTensor
    associations: Union[torch.FloatTensor, torch.IntTensor]  #because fair could be a float
    timing: torch.FloatTensor
    used_space: torch.BoolTensor

    def __getitem__(self, indices: torch.Tensor) -> Self:
        """
        Get the state at the specified indices

        Args:
            indices: torch.Tensor - Indices to get the state at
        Returns:
            RideshareState - State at the specified indices
        """
        return RideshareState(agents=self.agents,
                              locations=self.locations[indices],
                              destinations=self.destinations[indices],
                              associations=self.associations[indices],
                              timing=self.timing[indices],
                              used_space=self.used_space[indices])

    def log(self,
            path: str,
            new_episode: bool = False,
            constant_observations: Optional[List[str]] = [],
            initial: Optional[bool] = False,
            label: Optional[str] = None,
            partial_log: Optional[List[int]] = None,
            actions: Union[Dict[str, torch.Tensor], List[str]] = None,
            rewards: Union[Dict[str, torch.Tensor], List[float]] = None,
            infos: Dict[str, Any] = None,
            log_exclusions: List[str] = [],
            masked_attributes: Dict[Tuple[str, int], torch.Tensor] = None) -> None:
        """
        construct the logging version of the state

        Args:
            path: str - Path to save the log files (will split log files into seperate based on batches)
            new_episode: bool - is this a new episode?
            constant_observations: Optional[List[str] - list of attributes that are constant throughout the episode #!does nothing here
            initial: bool - use file headers 
            label: Optional[str] - a generic string filled in as the "label" column
            partial_log: Optional[List[int]] - list of specific batch indices to log #!does nothing here
            actions: Union[Dict[str, torch.Tensor], List[str]] - dictionary of actions to log
            exclusions: List[str] - list of attributes to exclude from logging #!does nothing here
            masked_attributes: Dict[Tuple[str, int], torch.Tensor] - dictionary of attributes (per batch) to mask rather than batch index #!does nothing here
        """

        if initial:
            try:
                os.mkdir(path)
            except FileExistsError:
                assert not os.path.exists(os.path.join(
                    path, "0.csv")), "path already exists and files found, check path. Don't waste experiments!"

        batch_size = torch.max((self.agents[:, 0] + 1).reshape(-1)).to(torch.int).item()

        if isinstance(actions, dict):
            # print(actions)
            actions = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in actions.items()}
        else:
            actions = {key: [None for _ in range(batch_size)] for key in actions}

        if isinstance(rewards, dict):
            rewards = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in rewards.items()}
        else:
            rewards = {key: [0.0 for _ in range(batch_size)] for key in rewards}


        if label is not None:
            assert "test" not in label and "train" not in label, "label should not be used to distinguish between test and train data make a new file"

        log_timing = self.timing[self.used_space].to('cpu')
        log_associations = self.associations[self.used_space].to('cpu')
        log_destinations = self.destinations[self.used_space].to('cpu')
        log_locations = self.locations[self.used_space].to('cpu')
        log_agents = self.agents.to('cpu')

        log_dict = {
            "timing": log_timing,
            "associations": log_associations,
            "destinations": log_destinations,
            "locations": log_locations,
        }

        log_dict = log_dict | {
            f"driver_{agent}_state": log_agents[log_agents[:, 1] == agent]
            for agent in torch.unique(log_agents[:, 1].to(torch.int)).tolist()
        }
        batches = torch.unique(log_agents[:, 0].to(torch.int)).tolist()

        for batch in batches:

            batch_random_variables = {key: [str(value[value[:, 0] == batch].tolist())] for key, value in log_dict.items()}
            batched_actions = {k: [v[batch]] for k, v in actions.items()}
            batched_rewards = {k + "_rewards": [v[batch]] for k, v in rewards.items()}

            batched_info = {}
            for _ag, _ag_infos in infos.items():
                for _info_key, _info_value in _ag_infos.items():
                    batched_info[f"{_ag}_{_info_key}"] = [v[batch] for v in _info_value]

            state_data = batch_random_variables | batched_actions | batched_rewards | batched_info

            df = pd.DataFrame(state_data)
            df['label'] = label
            df['new_episode'] = new_episode

            df.to_csv(os.path.join(path, f"{batch}.csv"), mode='a' if not initial else 'w', header=initial, index=False)

    def __hash__(self) -> int:
        """
        Get the hash of the state

        Returns:
            int - Hash of the state
        """
        #TODO use arange to check for unusued_space rather than hashing it

        keys = (self.agents, self.locations, self.destinations, self.associations, self.used_space)
        hashables = tuple([optimized_convert_hashable(key) for key in keys])
        return hash(hashables)
