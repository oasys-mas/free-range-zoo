"""Cybersecurity Environment Baseline Agents.

This module provides baseline agent implementations for the cybersecurity
defense environment. These agents can be used for benchmarking, debugging,
and as reference policies.

Available Baselines:
    NoopBaseline: Agent that always performs no-op actions.
    RandomBaseline: Agent that selects actions uniformly at random.
    CampDefenderBaseline: Defender that camps on a target node.
    PatchedAttackerBaseline: Attacker targeting patched nodes.
    PatchedDefenderBaseline: Defender targeting patched nodes.
    ExploitedAttackerBaseline: Attacker targeting exploited nodes.
    ExploitedDefenderBaseline: Defender targeting exploited nodes.

Example Usage:
    >>> from free_range_zoo.envs.cybersecurity.baselines import RandomBaseline
    >>> agent = RandomBaseline("attacker_0", parallel_envs=8)
    >>> actions = agent.act(action_space)
"""

from free_range_zoo.envs.cybersecurity.baselines.camp import CampDefenderBaseline
from free_range_zoo.envs.cybersecurity.baselines.noop import NoopBaseline
from free_range_zoo.envs.cybersecurity.baselines.random import RandomBaseline
from free_range_zoo.envs.cybersecurity.baselines.patched import PatchedAttackerBaseline, PatchedDefenderBaseline
from free_range_zoo.envs.cybersecurity.baselines.exploited import ExploitedAttackerBaseline, ExploitedDefenderBaseline

__all__ = [
    "CampDefenderBaseline",
    "NoopBaseline",
    "RandomBaseline",
    "PatchedAttackerBaseline",
    "PatchedDefenderBaseline",
    "ExploitedAttackerBaseline",
    "ExploitedDefenderBaseline",
]
