import unittest

from free_range_zoo.envs.wildfire.configs.aaai_2024 import aaai_2025_ol_config


class TestAAAI2024ConfigurationInitialization(unittest.TestCase):

    def test_aaai_2025_ol_config(self) -> None:
        aaai_2025_ol_config(1)
        aaai_2025_ol_config(2)
        aaai_2025_ol_config(3)
