"""Abstract class for environment configurations."""
from typing import Self
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class Configuration(ABC):
    """Abstract class for environment configurations."""

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate the configuration.

        Returns:
            bool - True if the configuration is valid
        """
        for attribute, value in self.__dict__.items():
            if hasattr(value, 'validate'):
                value.validate()

        return True

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

    def __post_init__(self):
        """Run the configuration validation on initialization."""
        self.validate()
