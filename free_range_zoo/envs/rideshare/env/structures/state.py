from typing import Self, Tuple, Union
from dataclasses import dataclass

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
