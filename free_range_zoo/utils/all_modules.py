"""Module Discovery and Environment Registry for FreeRangeZoo.

This module provides utilities for discovering and registering all
available FreeRangeZoo environments. It maintains a registry of
environment namespaces and individual environments.

Attributes:
    all_prefixes: List of registered namespace prefixes.
    manual_environments: Dictionary of manually controlled environments.
    oasys_mas: Dictionary of environments in the 'oasys_mas' namespace.
    all_environments: Nested dictionary of all environments by namespace.

Example Usage:
    >>> from free_range_zoo.utils.all_modules import all_environments
    >>> print(list(all_environments['oasys_mas'].keys()))
    ['wildfire_v0', 'rideshare_v0', 'cybersecurity_v0']
"""

from free_range_zoo.envs import wildfire_v0, rideshare_v0, cybersecurity_v0

#used in yaml_configs to find all configuration classes in frz
from free_range_zoo.envs.wildfire.env.v0.structures import configuration as w0conf
from free_range_zoo.envs.rideshare.env.v0.structures import configuration as r0conf
from free_range_zoo.envs.cybersecurity.env.v0.structures import configuration as c0conf

all_prefixes = ["oasys_mas"]

# environments which have manual policy scripts, allowing interactive play
manual_environments = {}

oasys_mas = {
    'oasys_mas/wildfire_v0': wildfire_v0,
    'oasys_mas/rideshare_v0': rideshare_v0,
    'oasys_mas/cybersecurity_v0': cybersecurity_v0
}

all_environments = {
    'oasys_mas': oasys_mas,
}
