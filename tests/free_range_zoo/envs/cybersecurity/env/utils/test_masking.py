from abc import ABC
import unittest
import torch

from free_range_zoo.envs.cybersecurity.env.utils.masking import mask_observation


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


class TestMasking(unittest.TestCase, TestCaching):

    def func(self, *args, **kwargs):
        return mask_observation(*args, **kwargs)

    def setUp(self):
        mask_observation.cache_clear()
        self.initial_args = ('attacker_1', True, True, True)
        self.different_args = ('attacker_1', False, True, True)

    @property
    def cache_info(self):
        return mask_observation.cache_info

    @property
    def cache_clear(self):
        return mask_observation.cache_clear

    def test_observe_other_power_masked_attacker(self) -> None:
        result = mask_observation('attacker_1', False, True, True)
        expected = torch.tensor([False, True])

        self.assertTrue(
            torch.equal(result, expected), f"""
            \rResult should be equal to expected
                \rExpected:\n{expected}
                \rActual:\n{result}""")

    def test_observe_other_power_masked_defender(self) -> None:
        result = mask_observation('defender_1', False, True, True)
        expected = torch.tensor([False, True, True])

        self.assertTrue(
            torch.equal(result, expected), f"""
            \rResult should be equal to expected
                \rExpected:\n{expected}
                \rActual:\n{result}""")

    def test_observe_other_presence_masked_attacker(self) -> None:
        result = mask_observation('attacker_1', True, False, True)
        expected = torch.tensor([True, False])

        self.assertTrue(
            torch.equal(result, expected), f"""
            \rResult should be equal to expected
                \rExpected:\n{expected}
                \rActual:\n{result}""")

    def test_observe_other_presence_masked_defender(self) -> None:
        result = mask_observation('defender_1', True, False, True)
        expected = torch.tensor([True, False, True])

        self.assertTrue(
            torch.equal(result, expected), f"""
            \rResult should be equal to expected
                \rExpected:\n{expected}
                \rActual:\n{result}""")

    def test_observe_other_location_masked_attacker(self) -> None:
        result = mask_observation('attacker_1', True, True, False)
        expected = torch.tensor([True, True])

        self.assertTrue(
            torch.equal(result, expected), f"""
            \rResult should be equal to expected
                \rExpected:\n{expected}
                \rActual:\n{result}""")

    def test_observe_other_location_masked_defender(self) -> None:
        result = mask_observation('defender_1', True, True, False)
        expected = torch.tensor([True, True, False])

        self.assertTrue(
            torch.equal(result, expected), f"""
            \rResult should be equal to expected
                \rExpected:\n{expected}
                \rActual:\n{result}""")
