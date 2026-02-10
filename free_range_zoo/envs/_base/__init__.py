"""Base Environment Components for FreeRangeZoo.

This module provides the foundational abstract classes and utilities for
implementing FreeRangeZoo environments. All environment implementations
should build upon these base classes.

Core Components:
    Abstract Base Classes:
        - Agent: Base class for agent implementations
        - Configuration: Base class for environment configurations
        - State: Base class for environment state representations
        - BatchedAECEnv (v0): Base class for batched AEC environments
        - BatchedEnv (v1): Base class for modern batched environments

    Utilities:
        - RandomGenerator: Deterministic random number generation
        - Logger: Base logging interface
        - CSVLogger: CSV file logging implementation
        - SQLLogger: SQL database logging implementation

    Versioned APIs:
        - v0: Original batched AEC environment API
        - v1: Modern batched environment API with dynamic agents

Structure:
    - agent.py: Agent interface
    - configuration.py: Configuration base class
    - state.py: State base class
    - random_generator.py: Random number generation
    - logging_handlers.py: Logging implementations
    - sql_logging.py: SQL logging models
    - v0/: Original environment API
    - v1/: Modern environment API

Example Usage:
    >>> from free_range_zoo.envs._base import Agent, Configuration, State
    >>> from free_range_zoo.envs._base.v0.env import BatchedAECEnv
    >>> from free_range_zoo.envs._base.v1.batched_env import BatchedEnv
"""

from free_range_zoo.envs._base.agent import Agent
from free_range_zoo.envs._base.configuration import Configuration
from free_range_zoo.envs._base.state import State
from free_range_zoo.envs._base.random_generator import RandomGenerator
from free_range_zoo.envs._base.logging_handlers import Logger, CSVLogger, SQLLogger

__all__ = [
    "Agent",
    "Configuration",
    "State",
    "RandomGenerator",
    "Logger",
    "CSVLogger",
    "SQLLogger",
]
