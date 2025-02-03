import unittest, os

from free_range_zoo_docs import build


class TestDocumentationBuilding(unittest.TestCase):

    @unittest.skipIf(
        os.name == 'nt',
        "This docbuild test fails on windows, but doc building works. Run `make html` in the docs directory in your conda environment to build the docs."
    )
    def test_build_does_not_raise_an_error(self):
        build()
