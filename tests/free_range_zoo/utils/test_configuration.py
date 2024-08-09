import unittest

from free_range_zoo.free_range_zoo.utils.configuration import Configuration


class TestConfiguration(unittest.TestCase):
    configuration = Configuration

    def test_configuration_includes_to(self) -> None:
        self.assertTrue(hasattr(self.configuration, 'to'),
                        f'{self.configuration.__name__} does not include to method')
        self.assertTrue(callable(self.configuration.to),
                        f'{self.configuration.__name__} to method is not callable')


if __name__ == '__main__':
    unittest.main()
