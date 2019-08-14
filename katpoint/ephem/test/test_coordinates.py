"""Tests fpor the coordinates module."""

import unittest

from ephem import degrees
from ephem import hours
from ephem import Galactic
from ephem import FixedBody

class TestGalactic(unittest.TestCase):
    """Test for the Galactic class."""
    def test_creation(self):
        """Test creation of Galactic objects."""
        l = degrees('10:10:40.123')
        b = degrees('40:20:50.567')
        gal = Galactic(l, b)
        lonlat = gal.get()
        self.assertEqual(str(lonlat[0]), '10:10:40.123')
        self.assertEqual(str(lonlat[1]), '40:20:50.567')

        ra = hours('10:10:40.123')
        dec = degrees('40:20:50.567')
        body = FixedBody()
        body._ra = ra
        body._dec = dec
        body.compute()
        gal = Galactic(body)
        lonlat = gal.get()
        self.assertEqual(str(lonlat[0]), '180:38:55.1466')
        self.assertEqual(str(lonlat[1]), '54:25:27.1913')

