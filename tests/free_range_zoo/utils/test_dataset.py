from typing import Optional, Dict, Any
import unittest
import torch

from free_range_zoo.utils.dataset import ConfigurationDataset, _deep_clone


def dummy_transform(config: Dict[str, Any], generator: Optional[torch.Generator]) -> Dict[str, Any]:
    """A simple transform that adds a key with a random value to the config."""
    config['random_value'] = torch.randint(0, 100, (1, ), generator=generator).item()
    return config


class TestConfigurationDataset(unittest.TestCase):

    def setUp(self):
        self.data = [{'data': i} for i in range(36)]
        self.transforms = [dummy_transform]
        self.seed = 0

    def test_splits_are_correctly_in_train_val_test(self) -> None:
        dataset = ConfigurationDataset(
            data=self.data,
            transforms=self.transforms,
            seed=self.seed,
            train_split=0.65,
            val_split=0.175,
            test_split=0.175,
        )

        train_len = len(dataset.train_data)
        val_len = len(dataset.val_data)
        test_len = len(dataset.test_data)

        self.assertEqual(train_len, 24)
        self.assertEqual(val_len, 6)
        self.assertEqual(test_len, 6)

    def test_transform_application(self) -> None:
        dataset = ConfigurationDataset(self.data, self.transforms, seed=self.seed)

        # Fetch first transformed item from train, val, and test
        train_item = dataset.train()
        self.assertIn('random_value', train_item)

        val_item = next(dataset.val())
        self.assertIn('random_value', val_item)

        test_item = next(dataset.test())
        self.assertIn('random_value', test_item)

    def test_val_set_consistency(self) -> None:
        dataset = ConfigurationDataset(self.data, self.transforms, seed=self.seed)

        val1 = [data for data in dataset.val()]
        val2 = [data for data in dataset.val()]

        # Validation sets should be identical
        self.assertEqual(len(val1), len(val2))
        for item1, item2 in zip(val1, val2):
            self.assertTrue(item1['random_value'], item2['random_value'])

    def test_test_set_consistency(self) -> None:
        dataset = ConfigurationDataset(self.data, self.transforms, seed=self.seed)

        test1 = [data for data in dataset.test()]
        test2 = [data for data in dataset.test()]

        # Validation sets should be identical
        self.assertEqual(len(test1), len(test2))
        for item1, item2 in zip(test1, test2):
            self.assertTrue(item1['random_value'], item2['random_value'])

    def test_train_shuffling(self) -> None:
        dataset = ConfigurationDataset(self.data, self.transforms, seed=self.seed)

        epoch1 = [dataset.train() for _ in range(len(dataset.train_data))]
        epoch2 = [dataset.train() for _ in range(len(dataset.train_data))]

        self.assertEqual(len(epoch1), len(epoch2))
        self.assertNotEqual([item['random_value'] for item in epoch1], [item['random_value'] for item in epoch2])


class TestDeepClone(unittest.TestCase):

    def test_deep_clone(self):
        """Test the _deep_clone function for correctness."""
        data = {'key': torch.tensor([1, 2, 3]), 'nested': {'key2': torch.tensor([4, 5, 6])}}
        cloned_data = _deep_clone(data)

        self.assertNotEqual(id(data), id(cloned_data))
        self.assertTrue(torch.equal(data['key'], cloned_data['key']))
        self.assertTrue(torch.equal(data['nested']['key2'], cloned_data['nested']['key2']))
