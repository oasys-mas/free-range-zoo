from abc import ABC
import itertools
import torch
import unittest

from free_range_rust import Space
from free_range_zoo.envs.wildfire.env.spaces.observations import (
    build_observation_space,
    build_single_observation_space,
    build_single_agent_observation_space,
    build_single_fire_observation_space,
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
        self.initial_args = (torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                           10]), 3, (10, 10, 5, 5), (10, 10, 3, 4), (True, True, True, True, True))

    def test_observation_space_structure(self) -> None:
        result = self.func(*self.initial_args)

        expected = [
            build_single_observation_space(self.initial_args[2], self.initial_args[3], i, self.initial_args[1],
                                           self.initial_args[4]) for i in range(11)
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
        self.agent_high = (10, 10, 5, 8, 4, 3, 2)
        self.fire_high = (10, 10, 3, 4)
        self.num_tasks = 2
        self.num_agents = 4
        self.initial_args = (
            self.agent_high,
            self.fire_high,
            self.num_tasks,
            self.num_agents,
            (True, True, True, True, True),
        )
        self.different_args = (
            self.agent_high,
            self.fire_high,
            self.num_tasks + 1,
            self.num_agents,
            (True, True, True, True, True),
        )

    def test_all_mask_combinations(self):
        """Test all feature masks for observation space masking."""
        for mask in itertools.product([True, False], repeat=5):
            with self.subTest(mask=mask):
                obs_space = build_single_observation_space(
                    self.agent_high,
                    self.fire_high,
                    self.num_tasks,
                    self.num_agents,
                    mask,
                )
                expected_mask = [True, True] + list(mask)

                masked_high = tuple(h for h, m in zip(self.agent_high, expected_mask) if m)
                for others_space in obs_space.spaces['others'].spaces:
                    self.assertEqual(
                        len(others_space.low),
                        len(masked_high),
                        f"Failed on mask {mask}, expected {len(masked_high)}, got {len(others_space.low)}",
                    )


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

        self.initial_args = ((10, 10, 5, 5), )
        self.different_args = ((10, 10, 5, 4), )

    def test_observation_space_structure(self) -> None:
        result = self.func(*self.initial_args)

        expected = Space.Box(low=[0] * 4, high=self.initial_args[0])

        self.assertEqual(result, expected, 'Observation space should match expected')


class TestBuildSingleFireObservationSpace(TestCaching, unittest.TestCase):

    def func(self, *args, **kwargs):
        return build_single_fire_observation_space(*args, **kwargs)

    @property
    def cache_info(self):
        return build_single_fire_observation_space.cache_info

    @property
    def cache_clear(self):
        return build_single_fire_observation_space.cache_clear

    def setUp(self) -> None:
        self.cache_clear()

        self.high = (10, 10, 5, 5)
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
