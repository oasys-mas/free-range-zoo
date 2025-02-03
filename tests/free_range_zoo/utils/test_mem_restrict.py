import unittest, os
from unittest.mock import patch, MagicMock

if os.name != 'nt':
    import resource
    from free_range_zoo.utils.mem_restrict import limit_memory


class TestLimitMemory(unittest.TestCase):

    @patch('psutil.virtual_memory')
    @patch('resource.setrlimit')
    @unittest.skipIf(os.name == 'nt', 'Resource module not available on Windows')
    def test_memory_limit(self, mock_setrlimit, mock_virtual_memory):
        mock_virtual_memory.return_value = MagicMock(available=1000000)

        #prevents import error even if the test is skipped
        if os.name != 'nt':
            limit_memory(0.1)
            mock_setrlimit.assert_called_once_with(resource.RLIMIT_AS, (100000, 100000))

    @patch('psutil.virtual_memory')
    @patch('resource.setrlimit')
    @unittest.skipIf(os.name == 'nt', 'Resource module not available on Windows')
    def test_memory_within_limit(self, mock_setrlimit, mock_virtual_memory):
        mock_virtual_memory.return_value = MagicMock(available=1000000)

        #prevents import error even if the test is skipped
        if os.name != 'nt':
            limit_memory(0.5)
            mock_setrlimit.assert_called_once_with(resource.RLIMIT_AS, (500000, 500000))


if __name__ == '__main__':
    unittest.main()
