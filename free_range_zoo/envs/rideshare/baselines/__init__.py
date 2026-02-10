"""Rideshare Environment Baseline Agents.

This module provides baseline agent implementations for the rideshare
coordination environment. These agents can be used for benchmarking,
debugging, and as reference policies.

Available Baselines:
    NoopBaseline: Agent that always performs no-op actions.
    RandomBaseline: Agent that selects actions uniformly at random.
    FirstInFirstOutTfocusBaseline: FIFO policy with task focus.
    FirstInFirstOutTglobalBaseline: FIFO policy with global focus.
    GreedyTaskFocus: Greedy policy with task focus.
    GreedyTaskGlobal: Greedy policy with global focus.

Example Usage:
    >>> from free_range_zoo.envs.rideshare.baselines import RandomBaseline
    >>> agent = RandomBaseline("driver_0", parallel_envs=8)
    >>> actions = agent.act(action_space)
"""

from free_range_zoo.envs.rideshare.baselines.noop import NoopBaseline
from free_range_zoo.envs.rideshare.baselines.random import RandomBaseline
from free_range_zoo.envs.rideshare.baselines.fifo_Tfocus import FirstInFirstOutTfocusBaseline
from free_range_zoo.envs.rideshare.baselines.fifo_Tglobal import FirstInFirstOutTglobalBaseline
from free_range_zoo.envs.rideshare.baselines.greedy_Tfocus import GreedyTaskFocus
from free_range_zoo.envs.rideshare.baselines.greedy_Tglobal import GreedyTaskGlobal
