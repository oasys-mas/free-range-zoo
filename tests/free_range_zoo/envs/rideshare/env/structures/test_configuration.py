import unittest

from abc import ABC

from free_range_zoo.envs.rideshare.env.structures.configuration import (
    AgentConfiguration,
    PassengerConfiguration,
    RideshareConfiguration,
    RewardConfiguration,
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


class TestPassengerConfiguration(TestConfiguration, unittest.TestCase):
    configuration = PassengerConfiguration


class TestAgentConfiguration(TestConfiguration, unittest.TestCase):
    configuration = AgentConfiguration


class TestRideshareConfiguration(TestConfiguration, unittest.TestCase):
    configuration = RideshareConfiguration


if __name__ == '__main__':
    unittest.main()
