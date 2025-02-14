import unittest
from abc import ABC

import torch

from free_range_rust import Space
from free_range_zoo.envs.rideshare.env.spaces.observations import (
    build_observation_space,
    build_single_observation_space,
    build_single_agent_observation_space,
    build_single_passenger_observation_space,
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


class TestBuildObservationSpace(unittest.TestCase):

    def func(self, *args, **kwargs):
        return build_observation_space(*args, **kwargs)

    def setUp(self) -> None:
        self.initial_args = (torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 4, (10, 10, 5, 8), (10, 10, 3, 3, 3, 3))

    def test_observation_space_structure(self) -> None:
        result = self.func(*self.initial_args)

        expected = [
            build_single_observation_space(*self.initial_args[2:4], i, self.initial_args[1], *self.initial_args[4:6])
            for i in range(11)
        ]

        self.assertEqual(result, expected, 'Observation spaces should match expected')


class TestBuildSingleObservationSpace(TestCaching, unittest.TestCase):

    def func(self, *args, **kwargs):
        return build_single_observation_space(*args, **kwargs)

    @property
    def cache_info(self):
        return build_single_observation_space.cache_info

    @property
    def cache_clear(self):
        return build_single_observation_space.cache_clear

    def setUp(self) -> None:
        self.cache_clear()

        self.initial_args = ((10, 10, 5, 8), (10, 10, 3, 3, 3, 3), 4, 3)
        self.different_args = ((10, 10, 5, 4), (10, 10, 3, 4, 3, 3), 3, 3)

    def test_observation_space_structure(self) -> None:
        result = self.func(*self.initial_args)

        expected = Space.Dict({
            'self':
            build_single_agent_observation_space(self.initial_args[0]),
            'others':
            Space.Tuple([*[build_single_agent_observation_space(self.initial_args[0]) for _ in range(2)]]),
            'tasks':
            build_single_passenger_observation_space(self.initial_args[1], self.initial_args[2])
        })

        self.assertEqual(result, expected, 'Observation space should match expected')

    def test_larger_agent_spaces(self) -> None:
        agent_spaces = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        for num_agents in agent_spaces:
            result = self.func(*self.initial_args[:3], num_agents, *self.initial_args[4:])

            expected = Space.Dict({
                'self':
                build_single_agent_observation_space(self.initial_args[0]),
                'others':
                Space.Tuple([*[build_single_agent_observation_space(self.initial_args[0]) for _ in range(num_agents - 1)]]),
                'tasks':
                build_single_passenger_observation_space(self.initial_args[1], self.initial_args[2])
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

        self.initial_args = ((10, 10, 4, 4), )
        self.different_args = ((10, 10, 5, 4), )

    def test_observation_space_structure(self) -> None:
        result = self.func(*self.initial_args)

        expected = Space.Box(low=[0] * 4, high=self.initial_args[0])

        self.assertEqual(result, expected, 'Observation space should match expected')


class TestBuildSinglePassengerObservationSpace(TestCaching, unittest.TestCase):

    def func(self, *args, **kwargs):
        return build_single_passenger_observation_space(*args, **kwargs)

    @property
    def cache_info(self):
        return build_single_passenger_observation_space.cache_info

    @property
    def cache_clear(self):
        return build_single_passenger_observation_space.cache_clear

    def setUp(self) -> None:
        self.cache_clear()

        self.high = (10, 5, 10, 5, 2, 2, 30, 5)
        self.initial_args = (self.high, 3)
        self.different_args = (self.high, 4)

    def test_observation_space_structure(self) -> None:
        task_spaces = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for num_tasks in task_spaces:
            result = self.func(self.high, num_tasks)

            expected = Space.Tuple([Space.Box(low=[0] * len(self.high), high=self.high) for _ in range(num_tasks)])

            self.assertEqual(result, expected, 'Observation space should match expected')


if __name__ == '__main__':
    unittest.main()
