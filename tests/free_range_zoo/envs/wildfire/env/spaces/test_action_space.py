import unittest
from abc import ABC

import torch

from free_range_rust import Space
from free_range_zoo.envs.wildfire.env.spaces.actions import build_single_action_space, build_action_space


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


class TestBuildActionSpace(unittest.TestCase):

    def func(self, *args, **kwargs):
        return build_action_space(*args, **kwargs)

    def test_build_environment_action_spaces(self) -> None:
        num_tasks_list = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        result = build_action_space(num_tasks_list)

        expected = Space.Vector([build_single_action_space(num_tasks) for num_tasks in num_tasks_list])

        self.assertEqual(result, expected, 'Action spaces should match expected')


class TestBuildSingleActionSpace(TestCaching, unittest.TestCase):

    def func(self, *args, **kwargs):
        return build_single_action_space(*args, **kwargs)

    @property
    def cache_info(self):
        return build_single_action_space.cache_info

    @property
    def cache_clear(self):
        return build_single_action_space.cache_clear

    def setUp(self) -> None:
        build_single_action_space.cache_clear()
        self.initial_args = (0, )
        self.different_args = (1, )

    def test_noop_element_included(self) -> None:
        result = build_single_action_space(0)

        expected = Space.OneOf([Space.Discrete(1, start=-1)])

        self.assertEqual(result, expected, 'Action space should match expected')

    def test_observation_space_structure(self) -> None:
        test_larger_action_spaces = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        for num_tasks in test_larger_action_spaces:
            result = build_single_action_space(num_tasks)

            expected = Space.OneOf([*[Space.Discrete(1, start=0) for _ in range(num_tasks)], Space.Discrete(1, start=-1)])

            self.assertEqual(result, expected, 'Action space should match expected')


if __name__ == '__main__':
    unittest.main()
