import unittest

from abc import ABC

from free_range_zoo.envs.wildfire.env.structures.configuration import (
    AgentConfiguration,
    FireConfiguration,
    StochasticConfiguration,
    WildfireConfiguration,
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


class TestFireConfiguration(TestConfiguration, unittest.TestCase):
    configuration = FireConfiguration


class TestAgentConfiguration(TestConfiguration, unittest.TestCase):
    configuration = AgentConfiguration


class TestRewardConfiguration(TestConfiguration, unittest.TestCase):
    configuration = RewardConfiguration


class TestStochasticConfiguration(TestConfiguration, unittest.TestCase):
    configuration = StochasticConfiguration


class TestWildfireConfiguration(TestConfiguration, unittest.TestCase):
    configuration = WildfireConfiguration


if __name__ == '__main__':
    unittest.main()
