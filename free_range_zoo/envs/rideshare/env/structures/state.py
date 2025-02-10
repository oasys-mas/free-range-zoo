"""State class representation for the rideshare state."""
from typing import Self
from dataclasses import dataclass
import torch
import pandas as pd
from free_range_zoo.utils.state import State
from free_range_zoo.utils.caching import optimized_convert_hashable


@dataclass
class RideshareState(State):
    """
    Representation of the rideshare state.

    Attributes:
        agents: torch.IntTensor - Locations of each agent in the form of <parallel_env, agent, (y, x)>
        passengers: torch.FloatTensor - Locations of each passenger in the form of <(batch, y, x, dest_x, dest_y, fare,
                                        state, association, entered_step, accepted_step, picked_step)>
    """

    agents: torch.IntTensor
    passengers: torch.FloatTensor

    def __getitem__(self, indices: torch.Tensor) -> Self:
        """
        Get the state at the specified indices.

        Args:
            indices: torch.Tensor - Indices to get the state at
        Returns:
            RideshareState - State at the specified indices
        """
        passenger_mask = self.passengers[:, 0].isin(indices)
        return RideshareState(
            agents=self.agents[indices],
            passengers=self.passengers[passenger_mask],
        )

    def __hash__(self) -> int:
        """
        Get the hash of the state.

        Returns:
            int - Hash of the state
        """
        keys = (self.agents, self.passengers)
        hashables = tuple([optimized_convert_hashable(key) for key in keys])
        return hash(hashables)

    def to_dataframe(self) -> int:
        """Convert the state into a dataframe."""
        shared = self.metadata.get('shared', ())
        blacklist = ('initial_state', 'checkpoint', 'metadata', 'passengers') + shared
        columns = [attribute for attribute in self.__dict__.keys() if attribute not in blacklist]
        data = {column: [str(row.tolist()) for row in self.__dict__[column]] for column in columns}

        passenger_data = []
        for i in range(self.agents.size(0)):
            passenger_data.append(str(self.passengers[self.passengers[:, 0] == i].tolist()))
        data['passengers'] = passenger_data

        df = pd.DataFrame(data)
        for key in shared:
            df[key] = str(self.__dict__[key].tolist())

        return df
