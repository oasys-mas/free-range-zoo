"""State representation for the cybersecurity environment."""

from typing import Self
from dataclasses import dataclass

import torch

from free_range_zoo.utils.state import State
from free_range_zoo.utils.caching import optimized_convert_hashable


@dataclass
class CybersecurityState(State):
    """
    Representation of the cybersecurity environment state.

    Subnetwork Attributes:
        network_state: torch.IntTensor - Graph data representing the subnetwork states and connections <Z, network, 1> (latency)
    Agent Attributes:
        location: torch.IntTensor - Tensor representing the current subnetwork position of each defender agent <Z, agent, 1> (location)
        presence: torch.BoolTensor - Tensor representing the presence of each all agents in the subnetworks (attackers, defenders) <Z, agent, 1> (presence)
    """

    network_state: torch.IntTensor

    location: torch.IntTensor
    presence: torch.BoolTensor

    def __getitem__(self, indices: torch.Tensor) -> Self:
        """
        Get the state at the specified indices.

        Args:
            indices: torch.Tensor - Indices to get the state at
        Returns:
            CybersecurityState - State at the specified indices
        """
        return CybersecurityState(
            network_state=self.network_state[indices],
            location=self.location[indices],
            presence=self.presence[indices],
        )

    def __hash__(self) -> int:
        """
        Get the hash of the state.

        Returns:
            int - Hash of the state
        """
        keys = (self.state, self.agents, self.presence)
        hashables = tuple([optimized_convert_hashable(key) for key in keys])
        return hash(hashables)
