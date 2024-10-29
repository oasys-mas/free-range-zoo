import unittest

from abc import ABC

from free_range_zoo.envs.cybersecurity.env.structures.configuration import (
    RewardConfiguration,
    AttackerConfiguration,
    DefenderConfiguration,
    NetworkConfiguration,
    StochasticConfiguration,
    CybersecurityConfiguration,
)


class TestConfiguration(ABC):
    configuration = None

    def test_configuration_includes_validate(self) -> None:
        self.assertTrue(hasattr(self.configuration, 'validate'),
                        f'{self.configuration.__name__} does not include validate method')
        self.assertTrue(callable(self.configuration.validate), f'{self.configuration.__name__} validate method is not callable')

    def test_configuration_includes_to(self) -> None:
        self.assertTrue(hasattr(self.configuration, 'to'), f'{self.configuration.__name__} does not include to method')
        self.assertTrue(callable(self.configuration.to), f'{self.configuration.__name__} to method is not callable')


class TestRewardConfiguration(TestConfiguration, unittest.TestCase):
    configuration = RewardConfiguration


class TestAttackerConfiguration(TestConfiguration, unittest.TestCase):
    configuration = AttackerConfiguration


class TestDefenderConfiguration(TestConfiguration, unittest.TestCase):
    configuration = DefenderConfiguration


class TestNetworkConfiguration(TestConfiguration, unittest.TestCase):
    configuration = NetworkConfiguration


class TestStochasticConfiguration(TestConfiguration, unittest.TestCase):
    configuration = StochasticConfiguration


class TestCybersecurityConfiguration(TestConfiguration, unittest.TestCase):
    configuration = CybersecurityConfiguration


if __name__ == '__main__':
    unittest.main()
