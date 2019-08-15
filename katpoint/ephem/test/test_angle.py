"""Tests for the angle module."""

import unittest

from ephem import degrees
from ephem import separation

class TestSeparation(unittest.TestCase):
    """Test for the separation method class."""
    def test_separation(self):
        """Test separation between two points."""
        a0 = degrees('10:10:40.123')
        a1 = degrees('40:20:50.567')
        b0 = degrees('10:11:40.123')
        b1 = degrees('41:20:50.567')
        s = separation((a0, a1), (b0, b1))
        self.assertEqual(str(s), '1:00:00.2861')

    def test_zero(self):
        """Test separation of the same point."""
        a0 = degrees('10:10:40.123')
        a1 = degrees('40:20:50.567')
        s = separation((a0, a1), (a0, a1))
        self.assertEqual(str(s), '0:00:00')
        self.assertEqual(s._a.radian, 0.0)
