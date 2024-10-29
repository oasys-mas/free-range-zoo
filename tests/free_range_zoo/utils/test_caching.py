from typing import Hashable
from abc import ABC
import unittest

import torch

from free_range_zoo.utils.caching import (convert_using_xxhash, convert_using_tuple, positional_encoding_hash,
                                          optimized_convert_hashable)


class TestHashableConversion(ABC):

    def setUp(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def func(self, data):
        raise NotImplementedError('Subclasses must implement this method')

    def test_convert_hashable(self) -> None:
        hashable = torch.tensor([1, 2, 3], device=self.device)
        converted = self.func(hashable)
        self.assertIsInstance(converted, Hashable, 'Converted hashable should be hashable')

    def test_convert_hashable_same(self) -> None:
        hashable = torch.tensor([1, 2, 3], device=self.device)
        converted = self.func(hashable)
        converted_again = self.func(hashable)
        self.assertEqual(converted, converted_again, 'Converted hashables should be equal')

    def test_convert_hashable_different(self) -> None:
        hashable = torch.tensor([1, 2, 3], device=self.device)
        hashable_different = torch.tensor([1, 2, 4], device=self.device)
        converted = self.func(hashable)
        converted_different = self.func(hashable_different)
        self.assertNotEqual(converted, converted_different, 'Converted hashables should not be equal')

    def test_hashable_is_not_tensor(self) -> None:
        hashable = torch.tensor([1, 2, 3], device=self.device)
        converted = self.func(hashable)

        self.assertNotIsInstance(converted, torch.Tensor, 'Converted hashable should not be a tensor')

    def test_final_datatype(self) -> None:
        if not hasattr(self, 'data_type'):
            return

        hashable = torch.tensor([1, 2, 3], device=self.device)
        converted = self.func(hashable)
        self.assertIsInstance(converted, self.data_type, 'Converted hashable should be of the correct data type')


class TestConvertUsingTuple(TestHashableConversion, unittest.TestCase):
    data_type = int

    def func(self, data):
        return convert_using_tuple(data)


class TestConvertUsingXxhash(TestHashableConversion, unittest.TestCase):
    data_type = int

    def func(self, data):
        return convert_using_xxhash(data)


@unittest.skipIf(not torch.cuda.is_available(), 'CUDA not available')
class TestConvertUsingGPUPositionalEncoding(TestHashableConversion, unittest.TestCase):
    data_type = float

    def func(self, data):
        return positional_encoding_hash(data)


class TestOptimizedConvertHashable(TestHashableConversion, unittest.TestCase):

    def func(self, data):
        return optimized_convert_hashable(data)


if __name__ == '__main__':
    unittest.main()
