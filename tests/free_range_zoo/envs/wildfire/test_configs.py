import unittest

from free_range_zoo.envs.wildfire.configs.planning import setup3, single_agent_two_fires, single_agent_single_fire
from free_range_zoo.envs.wildfire.configs.aaai_2024 import aaai_2025_ol_config


class TestPlanningConfigurationInitialization(unittest.TestCase):

    def test_setup3(self) -> None:
        setup3()

    def test_single_agent_two_fires(self) -> None:
        single_agent_two_fires()

    def test_single_agent_single_fire(self) -> None:
        single_agent_single_fire()


class TestAAAI2024ConfigurationInitialization(unittest.TestCase):

    def test_aaai_2025_ol_config(self) -> None:
        aaai_2025_ol_config(1)
        aaai_2025_ol_config(2)
        aaai_2025_ol_config(3)
