from typing import Self, Tuple
from dataclasses import dataclass

import torch

from free_range_zoo.free_range_zoo.utils.state import State
from free_range_zoo.free_range_zoo.utils.caching import optimized_convert_hashable


@dataclass
class WildfireState(State):
    """
    Representation of the wildfire environment state

    Fire Attributes:
        fires: torch.Tensor - Tensor representing the fire intensity of each cell
        intensity: torch.Tensor - Tensor representing the intensity of each fire
        fuel: torch.Tensor - Tensor representing the fuel remaining in each cell
    Agent Attributes:
        agents: torch.Tensor - Tensor representing the location of each agent
        suppressants: torch.Tensor - Tensor representing the suppressant of each agent
        capacity: torch.Tensor - Tensor representing the maximum suppressant of each agent
        equipment: torch.Tensor - Tensor representing the power of each agent
    """
    fires: torch.Tensor
    intensity: torch.Tensor
    fuel: torch.Tensor

    agents: torch.Tensor
    suppressants: torch.Tensor
    capacity: torch.Tensor
    equipment: torch.Tensor

    def __getitem__(self, indices: torch.Tensor) -> Self:
        """
        Get the state at the specified indices

        Args:
            indices: torch.Tensor - Indices to get the state at
        Returns:
            WildfireState - State at the specified indices
        """
        return WildfireState(fires=self.fires[indices],
                             intensity=self.intensity[indices],
                             fuel=self.fuel[indices],
                             agents=self.agents,
                             suppressants=self.suppressants[indices],
                             capacity=self.capacity[indices],
                             equipment=self.equipment[indices])

    def __hash__(self) -> int:
        """
        Get the hash of the state

        Returns:
            int - Hash of the state
        """
        keys = (self.fires, self.intensity, self.fuel, self.agents, self.suppressants, self.capacity, self.equipment)
        hashables = tuple([optimized_convert_hashable(key) for key in keys])
        return hash(hashables)
