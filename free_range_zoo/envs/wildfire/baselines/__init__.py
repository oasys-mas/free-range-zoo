"""Wildfire Environment Baseline Agents.

This module provides baseline agent implementations for the wildfire
suppression environment. These agents can be used for benchmarking,
debugging, and as reference policies.

Available Baselines:
    NoopBaseline: Agent that always performs no-op actions.
    RandomBaseline: Agent that selects actions uniformly at random.
    StrongestBaseline: Agent that targets the strongest fire.
    WeakestBaseline: Agent that targets the weakest fire.

Example Usage:
    >>> from free_range_zoo.envs.wildfire.baselines import NoopBaseline
    >>> agent = NoopBaseline("firefighter_0", parallel_envs=8)
    >>> actions = agent.act(action_space)
"""

from free_range_zoo.envs.wildfire.baselines.noop import NoopBaseline
from free_range_zoo.envs.wildfire.baselines.random import RandomBaseline
from free_range_zoo.envs.wildfire.baselines.strongest import StrongestBaseline
from free_range_zoo.envs.wildfire.baselines.weakest import WeakestBaseline
