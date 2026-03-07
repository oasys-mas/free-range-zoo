"""Abstract Configuration Base Class for FreeRangeZoo Environments.

This module provides the Configuration abstract base class that all
environment configuration classes must inherit from. Configurations
define the parameters and constraints for environment initialization.

Classes:
    Configuration: Abstract base class for environment configurations.

Example Usage:
    >>> from free_range_zoo.envs._base.configuration import Configuration
    >>> class MyConfig(Configuration):
    ...     def validate(self) -> bool:
    ...         # Validate configuration parameters
    ...         pass
"""

from typing import Self
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class Configuration(ABC):
    """Abstract class for environment configurations."""
    _subclass_registry = {}

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate the configuration.

        Returns:
            bool: Whether the configuration is valid
        """
        for attribute, value in self.__dict__.items():
            if hasattr(value, 'validate'):
                value.validate()

        return True

    @classmethod
    def type_caster(cls):
        """
        Default type caster for use in yaml config reading. 
        Subclasses can override this to provide custom type casting.

        Returns:
            dict - A dictionary of callables by input (config_name, field_name)
        """
        return {} if not hasattr(cls, '_type_caster') else cls._type_caster

    def __init_subclass__(cls) -> None:
        Configuration._subclass_registry[f'{cls.__module__}:{cls.__name__}'] = cls
        return super().__init_subclass__()

    @classmethod
    def get_subclasses(cls):
        """
        Return all environment instances of configurations
        Returns:
            dict - A dictionary of environment names to their configuration classes
        """
        return Configuration._subclass_registry

    def to(self, device: torch.DeviceObjType = torch.device('cpu')) -> Self:
        """
        Move all tensors to the specified device.

        Args:
            device (torch.DeviceObjType): Device to move tensors to.

        Returns:
            Self: The modified configuration.
        """
        for attribute, value in self.__dict__.items():
            if hasattr(value, 'to'):
                setattr(self, attribute, value.to(device))

        return self

    def __post_init__(self):
        """Run the configuration validation on initialization."""
        self.validate()
