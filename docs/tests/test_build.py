import unittest

from free_range_zoo_docs import build


class TestDocumentationBuilding(unittest.TestCase):

    def test_build_does_not_raise_an_error(self):
        build()
