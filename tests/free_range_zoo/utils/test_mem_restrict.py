import unittest
from unittest.mock import patch, MagicMock

import resource
from free_range_zoo.utils.mem_restrict import limit_memory


class TestLimitMemory(unittest.TestCase):

    @patch('psutil.virtual_memory')
    @patch('resource.setrlimit')
    def test_memory_limit(self, mock_setrlimit, mock_virtual_memory):
        mock_virtual_memory.return_value = MagicMock(available=1000000)

        limit_memory(0.1)

        mock_setrlimit.assert_called_once_with(resource.RLIMIT_AS, (100000, 100000))

    @patch('psutil.virtual_memory')
    @patch('resource.setrlimit')
    def test_memory_within_limit(self, mock_setrlimit, mock_virtual_memory):
        mock_virtual_memory.return_value = MagicMock(available=1000000)

        limit_memory(0.5)

        mock_setrlimit.assert_called_once_with(resource.RLIMIT_AS, (500000, 500000))


if __name__ == '__main__':
    unittest.main()
