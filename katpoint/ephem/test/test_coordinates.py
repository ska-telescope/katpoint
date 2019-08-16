"""Tests for the coordinates module."""

import unittest

from ephem import degrees
from ephem import hours
from ephem import Galactic
from ephem import Equatorial
from ephem import FixedBody
from ephem import Observer

class TestGalactic(unittest.TestCase):
    """Test for the Galactic class."""
    def test_creation(self):
        """Test creation of Galactic objects."""
        l = degrees('10:10:40.123')
        b = degrees('40:20:50.567')
        gal = Galactic(l, b)
        lonlat = gal.get()
        self.assertAlmostEqual(lonlat[0], 0.177636329)
        self.assertAlmostEqual(lonlat[1], 0.704194621)
        self.assertEqual(str(lonlat[0]), '10:10:40.123')
        self.assertEqual(str(lonlat[1]), '40:20:50.567')

        ra = hours('10:10:40.123')
        dec = degrees('40:20:50.567')
        body = FixedBody()
        body._ra = ra
        body._dec = dec
        body.compute(Observer())
        gal = Galactic(body)
        lonlat = gal.get()
        self.assertEqual(str(lonlat[0]), '180:38:55.1735')
        self.assertEqual(str(lonlat[1]), '54:25:27.2027')

    def test_to_radec(self):
        l = degrees('10:10:40.123')
        b = degrees('40:20:50.567')
        gal = Galactic(l, b)
        radec = gal.to_radec()
        self.assertAlmostEqual(radec[0], 15.0 * 0.276394626)
        self.assertAlmostEqual(radec[1], 0.032872188)
        self.assertEqual(str(radec[0]), '15:50:10.484')
        self.assertEqual(str(radec[1]), '1:53:00.3754')


class TestEquatorial(unittest.TestCase):
    """Test for the Equatorial class."""
    def test_creation(self):
        ra = hours('10:10:40.123')
        dec = degrees('40:20:50.567')
        equ = Equatorial(ra, dec)
        radec = equ.get()
        self.assertAlmostEqual(radec[0], 15.0 * 0.177636329)
        self.assertAlmostEqual(radec[1], 0.704194621)
        self.assertEqual(str(radec[0]), '10:10:40.123')
        self.assertEqual(str(radec[1]), '40:20:50.567')

        equ2 = Equatorial(equ)
        radec = equ2.get()
        self.assertAlmostEqual(radec[0], 15.0 * 0.177636329)
        self.assertAlmostEqual(radec[1], 0.704194621)
        self.assertEqual(str(radec[0]), '10:10:40.123')
        self.assertEqual(str(radec[1]), '40:20:50.567')

        l = degrees('10:10:40.123')
        b = degrees('40:20:50.567')
        gal = Galactic(l, b)
        equ3 = Equatorial(gal)
        radec = equ3.get()
        self.assertAlmostEqual(radec[0], 15.0 * 0.276394626)
        self.assertAlmostEqual(radec[1], 0.032872188)
        self.assertEqual(str(radec[0]), '15:50:10.484')
        self.assertEqual(str(radec[1]), '1:53:00.3754')
