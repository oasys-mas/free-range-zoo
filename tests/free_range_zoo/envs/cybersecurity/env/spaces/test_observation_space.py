import unittest
from abc import ABC

import torch
import numpy as np
from gymnasium.spaces import Box, Tuple, Dict

from free_range_zoo.free_range_zoo.envs.cybersecurity.env.spaces.observations import (
    build_observation_space,
    build_single_agent_observation_space,
    build_single_defender_observation_space,
    build_single_attacker_observation_space,
    build_single_subnetwork_observation_space,
)


class TestCaching(ABC):

    def func(self, *args, **kwargs):
        raise NotImplementedError('Subclasses must implement this method')

    def setUp(self):
        self.func.cache_clear()

    def test_cache_inittial_miss(self):
        self.func(*self.initial_args)
        self.assertEqual(self.cache_info().hits, 0, 'Cache should not have been hit')
        self.assertEqual(self.cache_info().misses, 1, 'Cache should have been missed')

    def test_cache_hit_after_miss(self):
        self.func(*self.initial_args)
        self.func(*self.initial_args)
        self.assertEqual(self.cache_info().hits, 1, 'Cache should have been hit')
        self.assertEqual(self.cache_info().misses, 1, 'Cache should not have been missed')

    def test_cache_miss_after_different_args(self):
        self.func(*self.initial_args)
        self.func(*self.different_args)
        self.assertEqual(self.cache_info().hits, 0, 'Cache should not have been hit')
        self.assertEqual(self.cache_info().misses, 2, 'Cache should have been missed')

    def test_cache_hit_with_previous_args(self):
        self.func(*self.initial_args)
        self.func(*self.different_args)
        self.func(*self.initial_args)
        self.assertEqual(self.cache_info().hits, 1, 'Cache should have been hit')
        self.assertEqual(self.cache_info().misses, 2, 'Cache should not have been missed')


class TestBuildObservationSpace(TestCaching, unittest.TestCase):

    def func(self, *args, **kwargs):
        return build_observation_space(*args, **kwargs)

    @property
    def cache_info(self):
        return build_observation_space.cache_info

    @property
    def cache_clear(self):
        return build_observation_space.cache_clear

    def setUp(self) -> None:
        self.cache_clear()

        self.initial_args = ('attacker', 3, 10, 5, 5, (10, 1), (3, 2, 1), (5, ), True, True, True)
        self.different_args = ('defender', 3, 10, 5, 5, (10, 1), (3, 2, 1), (5, ), True, True, True)

    def test_attacker_space_structure(self) -> None:
        result = self.func(*self.initial_args)

        expected = [
            build_single_attacker_observation_space(self.initial_args[5], self.initial_args[7], self.initial_args[1],
                                                    self.initial_args[3], self.initial_args[8], self.initial_args[9])
        ] * 10

        self.assertEqual(result, expected, 'Observation space should match expected')

    def test_defender_observation_space_structure(self) -> None:
        result = self.func(*self.different_args)
        expected = [
            build_single_defender_observation_space(self.different_args[6], self.different_args[7], self.different_args[1],
                                                    self.different_args[4], self.different_args[8], self.different_args[9],
                                                    self.different_args[10])
        ] * 10

        self.assertEqual(result, expected, 'Observation space should match expected')


class TestSingleAttackerObservationSpace(TestCaching, unittest.TestCase):

    def func(self, *args, **kwargs):
        return build_single_attacker_observation_space(*args, **kwargs)

    @property
    def cache_info(self):
        return build_single_attacker_observation_space.cache_info

    @property
    def cache_clear(self):
        return build_single_attacker_observation_space.cache_clear

    def setUp(self) -> None:
        self.cache_clear()

        self.initial_args = ((10, 1), (3, ), 1, 3, True, True)
        self.different_args = ((10, 0), (3, ), 1, 3, True, True)

    def test_observation_space_structure(self) -> None:
        result = self.func(*self.initial_args)

        expected = Dict({
            'self': build_single_agent_observation_space(self.initial_args[0]),
            'others': Tuple([build_single_agent_observation_space(self.initial_args[0]) for _ in range(2)]),
            'tasks': build_single_subnetwork_observation_space(self.initial_args[1], self.initial_args[2]),
        })

        self.assertEqual(result, expected, 'Observation space should match expected')

    def test_excludes_power(self) -> None:
        result = self.func(*self.initial_args[:-2], False, True)
        expected = Dict({
            'self': build_single_agent_observation_space(self.initial_args[0]),
            'others': Tuple([build_single_agent_observation_space(self.initial_args[0][1:]) for _ in range(2)]),
            'tasks': build_single_subnetwork_observation_space(self.initial_args[1], self.initial_args[2]),
        })
        self.assertEqual(result, expected, 'Observation space should match expected')

    def test_excludes_presence(self) -> None:
        result = self.func(*self.initial_args[:-2], True, False)
        expected = Dict({
            'self': build_single_agent_observation_space(self.initial_args[0]),
            'others': Tuple([build_single_agent_observation_space(self.initial_args[0][:1]) for _ in range(2)]),
            'tasks': build_single_subnetwork_observation_space(self.initial_args[1], self.initial_args[2]),
        })
        self.assertEqual(result, expected, 'Observation space should match expected')

    def test_excludes_all(self) -> None:
        result = self.func(*self.initial_args[:-2], False, False)
        expected = Dict({
            'self': build_single_agent_observation_space(self.initial_args[0]),
            'others': Tuple([build_single_agent_observation_space(()) for _ in range(2)]),
            'tasks': build_single_subnetwork_observation_space(self.initial_args[1], self.initial_args[2]),
        })
        self.assertEqual(result, expected, 'Observation space should match expected')


class TestBuildSingleDefenderObservationSpace(TestCaching, unittest.TestCase):

    def func(self, *args, **kwargs):
        return build_single_defender_observation_space(*args, **kwargs)

    @property
    def cache_info(self):
        return build_single_defender_observation_space.cache_info

    @property
    def cache_clear(self):
        return build_single_defender_observation_space.cache_clear

    def setUp(self) -> None:
        self.cache_clear()

        self.initial_args = ((10, 1, 5), (3, ), 1, 3, True, True, True)
        self.different_args = ((10, 1, 6), (3, ), 3, 3, True, True, True)

    def test_observation_space_structure(self) -> None:
        result = self.func(*self.initial_args)

        expected = Dict({
            'self': build_single_agent_observation_space(self.initial_args[0]),
            'others': Tuple([build_single_agent_observation_space(self.initial_args[0]) for _ in range(2)]),
            'tasks': build_single_subnetwork_observation_space(self.initial_args[1], self.initial_args[2]),
        })

        self.assertEqual(result, expected, 'Observation space should match expected')

    def test_excludes_power(self) -> None:
        result = self.func(*self.initial_args[:-3], False, True, True)
        expected = Dict({
            'self': build_single_agent_observation_space(self.initial_args[0]),
            'others': Tuple([build_single_agent_observation_space(self.initial_args[0][1:]) for _ in range(2)]),
            'tasks': build_single_subnetwork_observation_space(self.initial_args[1], self.initial_args[2]),
        })
        self.assertEqual(result, expected, 'Observation space should match expected')

    def test_excludes_presence(self) -> None:
        result = self.func(*self.initial_args[:-3], True, False, True)
        expected = Dict({
            'self':
            build_single_agent_observation_space(self.initial_args[0]),
            'others':
            Tuple([build_single_agent_observation_space((self.initial_args[0][0], self.initial_args[0][2])) for _ in range(2)]),
            'tasks':
            build_single_subnetwork_observation_space(self.initial_args[1], self.initial_args[2]),
        })
        self.assertEqual(result, expected, 'Observation space should match expected')

    def test_excludes_location(self) -> None:
        result = self.func(*self.initial_args[:-3], True, True, False)
        expected = Dict({
            'self': build_single_agent_observation_space(self.initial_args[0]),
            'others': Tuple([build_single_agent_observation_space(self.initial_args[0][:-1]) for _ in range(2)]),
            'tasks': build_single_subnetwork_observation_space(self.initial_args[1], self.initial_args[2]),
        })
        self.assertEqual(result, expected, 'Observation space should match expected')

    def test_excludes_all(self) -> None:
        result = self.func(*self.initial_args[:-3], False, False, False)
        expected = Dict({
            'self': build_single_agent_observation_space(self.initial_args[0]),
            'others': Tuple([build_single_agent_observation_space(()) for _ in range(2)]),
            'tasks': build_single_subnetwork_observation_space(self.initial_args[1], self.initial_args[2]),
        })

        self.assertEqual(result, expected, 'Observation space should match expected')


class TestBuildSingleAgentObservationSpace(TestCaching, unittest.TestCase):

    def func(self, *args, **kwargs):
        return build_single_agent_observation_space(*args, **kwargs)

    @property
    def cache_info(self):
        return build_single_agent_observation_space.cache_info

    @property
    def cache_clear(self):
        return build_single_agent_observation_space.cache_clear

    def setUp(self) -> None:
        self.cache_clear()

        self.high = (10, 1, 5)
        self.initial_args = ((9, 1, 5), )
        self.different_args = ((10, 1, 5), )

    def test_observation_space_structure(self) -> None:
        result = self.func(self.high)

        expected = Box(low=np.array([0] * len(self.high)), high=np.array(self.high))

        self.assertEqual(result, expected, 'Observation space should match expected')


class TestBuildSingleSubnetworkObservationSpace(TestCaching, unittest.TestCase):

    def func(self, *args, **kwargs):
        return build_single_subnetwork_observation_space(*args, **kwargs)

    @property
    def cache_info(self):
        return build_single_subnetwork_observation_space.cache_info

    @property
    def cache_clear(self):
        return build_single_subnetwork_observation_space.cache_clear

    def setUp(self) -> None:
        self.cache_clear()

        self.high = (5, )
        self.initial_args = (self.high, 3)
        self.different_args = (self.high, 4)

    def test_observation_space_structure(self) -> None:
        tasks = torch.arange(1, 11)

        for num_tasks in tasks:
            result = self.func(self.high, num_tasks)

            expected = Tuple([Box(low=np.array([0] * len(self.high)), high=np.array(self.high)) for _ in range(num_tasks)])

            self.assertEqual(result, expected, 'Observation space should match expected')


if __name__ == '__main__':
    unittest.main()
