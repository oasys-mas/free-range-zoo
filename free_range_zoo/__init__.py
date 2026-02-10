"""FreeRangeZoo: A Library of Multi-Agent Reinforcement Learning Environments.

This package provides a collection of diverse multi-agent environments for
training and evaluating reinforcement learning agents. It includes environments
for wildfire suppression, rideshare optimization, and cybersecurity defense.

Available Environments:
    - wildfire_v0: Wildfire suppression simulation with multiple firefighters
    - rideshare_v0: Rideshare coordination with driver and passenger agents
    - cybersecurity_v0: Network defense scenario with attackers and defenders

Key Features:
    - PettingZoo-compatible API (AEC and Parallel environments)
    - Dynamic agent populations (agents can enter/leave during episodes)
    - Dynamic task generation (tasks appear/disappear based on environment state)
    - GPU-accelerated simulations using PyTorch
    - Built-in baseline agents for benchmarking
    - Comprehensive logging and replay capabilities

Quick Start:
    >>> from free_range_zoo.envs import wildfire_v0, rideshare_v0, cybersecurity_v0
    >>> env = wildfire_v0.env()
    >>> observations, infos = env.reset()
"""

from free_range_zoo.envs import (
    wildfire_v0,
    rideshare_v0,
    cybersecurity_v0,
)

__version__ = "0.1.0"

__all__ = [
    "wildfire_v0",
    "rideshare_v0",
    "cybersecurity_v0",
]