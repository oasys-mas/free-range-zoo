from typing import Hashable
from abc import ABC
import unittest

import torch

from free_range_zoo.free_range_zoo.utils.caching import (convert_using_xxhash,
                                                         convert_using_tuple,
                                                         optimized_convert_hashable)


class TestHashableConversion(ABC):
    def func(self, data):
        raise NotImplementedError('Subclasses must implement this method')

    def test_convert_hashable(self) -> None:
        hashable = torch.tensor([1, 2, 3])
        converted = self.func(hashable)
        self.assertIsInstance(converted, Hashable, 'Converted hashable should be hashable')

    def test_convert_hashable_same(self) -> None:
        hashable = torch.tensor([1, 2, 3])
        converted = self.func(hashable)
        converted_again = self.func(hashable)
        self.assertEqual(converted, converted_again, 'Converted hashables should be equal')

    def test_convert_hashable_different(self) -> None:
        hashable = torch.tensor([1, 2, 3])
        hashable_different = torch.tensor([1, 2, 4])
        converted = self.func(hashable)
        converted_different = self.func(hashable_different)
        self.assertNotEqual(converted, converted_different, 'Converted hashables should not be equal')


class TestConvertUsingTuple(TestHashableConversion, unittest.TestCase):
    def func(data):
        return convert_using_tuple(data)


class TestConvertUsingXxhash(TestHashableConversion, unittest.TestCase):
    def func(self, data):
        return convert_using_xxhash(data)


class TestOptimizedConvertHashable(TestHashableConversion, unittest.TestCase):
    def func(self, data):
        return optimized_convert_hashable(data)


if __name__ == '__main__':
    unittest.main()
