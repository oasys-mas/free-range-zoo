"""Dataset class for generating environment configurations and splitting configs into training, validation, and test sets."""
from typing import List, Callable, Dict, Any, Optional
import torch
import itertools
import copy


def _deep_clone(data):
    if isinstance(data, torch.Tensor):
        return data.clone()
    elif isinstance(data, list):
        return [_deep_clone(item) for item in data]
    elif isinstance(data, dict):
        return {key: _deep_clone(value) for key, value in data.items()}
    else:
        return copy.deepcopy(data)


class ConfigurationDataset:
    """A dataset that splits data into training, validation, and test sets."""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        transforms: List[Callable[[Dict[str, Any], Optional[torch.Generator]], int]],
        seed: Optional[int] = None,
        train_split: Optional[float] = 0.66,
        val_split: Optional[float] = 0.17,
        test_split: Optional[float] = 0.17,
        val_seed: Optional[int] = 10,
        test_seed: Optional[int] = 20,
    ):
        """
        Create a new ConfigurationDataset.

        Args:
            data: The data to split.
            transforms: A list of transforms to apply to the data.
            seed: The seed to use for the random number generator.
            train_split: The proportion of the data to use for training.
            val_split: The proportion of the data to use for validation.
            test_split: The proportion of the data to use for testing.
        """
        for transform in transforms:
            assert callable(transform), "All transforms must be callable"
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "The splits must sum to 1.0"

        for datum in data:
            assert isinstance(datum, dict), "All data must be dictionaries"

        instances = len(data)
        val_count = int(val_split * instances)
        test_count = int(test_split * instances)
        train_count = instances - val_count - test_count

        # Assign data using the pattern
        split_pattern = ['train'] * train_count + ['val'] * val_count + ['test'] * test_count
        split_cycle = itertools.cycle(split_pattern)

        self.train_data = []
        self.val_data = []
        self.test_data = []

        for datum in data:
            match next(split_cycle):
                case 'train':
                    self.train_data.append(datum)
                case 'val':
                    self.val_data.append(datum)
                case 'test':
                    self.test_data.append(datum)

        self.transforms = transforms

        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1, )).item()
        self.generator = torch.Generator().manual_seed(seed)

        self.val_seed = val_seed
        self.test_seed = test_seed

        self.train_iterator = self._train_generator()

    @torch.no_grad()
    def train(self):
        """Return the next configuration instance from the infinite train dataset."""
        return next(self.train_iterator)

    @torch.no_grad()
    def val(self):
        """Return an iterator over the validation dataset."""
        generator = torch.Generator().manual_seed(self.val_seed)
        for data_item in self.val_data:
            yield self._apply_transforms(_deep_clone(data_item), generator)

    @torch.no_grad()
    def test(self):
        """Return an iterator over the test dataset."""
        generator = torch.Generator().manual_seed(self.test_seed)
        for data_item in self.test_data:
            yield self._apply_transforms(_deep_clone(data_item), generator)

    @torch.no_grad()
    def _apply_transforms(self, data_item: Dict[str, Any], generator: torch.Generator):
        transformed_item = data_item
        for transform in self.transforms:
            transformed_item = transform(transformed_item, generator)
        return transformed_item

    @torch.no_grad()
    def _train_generator(self):
        while True:
            indices = torch.randperm(len(self.train_data), generator=self.generator)
            for idx in indices:
                datum = self.train_data[idx]
                yield self._apply_transforms(_deep_clone(datum), self.generator)
