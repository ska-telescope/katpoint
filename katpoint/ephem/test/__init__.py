"""Unit test suite for katpoint emulation of pyephem."""

import logging
import sys
import unittest

from ephem.test import test_coordinates

def suite():
    loader = unittest.TestLoader()
    testsuite = unittest.TestSuite()
    testsuite.addTests(loader.loadTestsFromModule(test_coordinates))
    return testsuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
