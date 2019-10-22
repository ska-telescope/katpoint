"""Unit test suite for katpoint emulation of pyephem."""

import logging
import sys
import unittest

from ephem.test import test_angle
from ephem.test import test_coordinates
from ephem.test import test_body
from ephem.test import test_observer

def suite():
    loader = unittest.TestLoader()
    testsuite = unittest.TestSuite()
    testsuite.addTests(loader.loadTestsFromModule(test_coordinates))
    return testsuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
