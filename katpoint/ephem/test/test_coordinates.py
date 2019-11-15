"""Tests for the coordinates module."""

import unittest
from astropy import coordinates
from astropy import units

from ephem import Galactic
from ephem import Equatorial
from ephem import FixedBody
from ephem import Observer

class TestGalactic(unittest.TestCase):
    """Test for the Galactic class."""
    def test_creation(self):
        """Test creation of Galactic objects."""
        l = coordinates.Longitude('10:10:40.123', unit=units.deg)
        b = coordinates.Latitude('40:20:50.567', unit=units.deg)
        gal = Galactic(l, b)
        lonlat = gal.get()
        self.assertAlmostEqual(lonlat[0].rad, 0.177636329)
        self.assertAlmostEqual(lonlat[1].rad, 0.704194621)
        self.assertEqual(lonlat[0].to_string(sep=':'), '10:10:40.123')
        self.assertEqual(lonlat[1].to_string(sep=':'), '40:20:50.567')

        ra = coordinates.Longitude('10:10:40.123', unit=units.hour)
        dec = coordinates.Latitude('40:20:50.567', unit=units.deg)
        body = FixedBody()
        body._ra = ra
        body._dec = dec
        body.compute(Observer())
        gal = Galactic(body)
        lonlat = gal.get()
        self.assertEqual(lonlat[0].to_string(sep=':'), '180:38:55.1735')
        self.assertEqual(lonlat[1].to_string(sep=':'), '54:25:27.2027')

    def test_to_radec(self):
        l = coordinates.Longitude('10:10:40.123', unit=units.deg)
        b = coordinates.Latitude('40:20:50.567', unit=units.deg)
        gal = Galactic(l, b)
        radec = gal.to_radec()
        self.assertAlmostEqual(radec[0].rad, 15.0 * 0.276394626)
        self.assertAlmostEqual(radec[1].rad, 0.032872188)
        self.assertEqual(radec[0].to_string(sep=':', unit=units.hour),
                '15:50:10.484')
        self.assertEqual(radec[1].to_string(sep=':'), '1:53:00.3754')


class TestEquatorial(unittest.TestCase):
    """Test for the Equatorial class."""
    def test_creation(self):
        ra = coordinates.Longitude('10:10:40.123', unit=units.hour)
        dec = coordinates.Latitude('40:20:50.567', unit=units.deg)
        equ = Equatorial(ra, dec)
        radec = equ.get()
        self.assertAlmostEqual(radec[0].rad, 15.0 * 0.177636329)
        self.assertAlmostEqual(radec[1].rad, 0.704194621)
        self.assertEqual(radec[0].to_string(sep=':', unit=units.hour),
                '10:10:40.123')
        self.assertEqual(radec[1].to_string(sep=':'), '40:20:50.567')

        equ2 = Equatorial(equ)
        radec = equ2.get()
        self.assertAlmostEqual(radec[0].rad, 15.0 * 0.177636329)
        self.assertAlmostEqual(radec[1].rad, 0.704194621)
        self.assertEqual(radec[0].to_string(sep=':', unit=units.hour),
                '10:10:40.123')
        self.assertEqual(radec[1].to_string(sep=':'), '40:20:50.567')

        l = coordinates.Longitude('10:10:40.123', unit=units.deg)
        b = coordinates.Latitude('40:20:50.567', unit=units.deg)
        gal = Galactic(l, b)
        equ3 = Equatorial(gal)
        radec = equ3.get()
        self.assertAlmostEqual(radec[0].rad, 15.0 * 0.276394626)
        self.assertAlmostEqual(radec[1].rad, 0.032872188)
        self.assertEqual(radec[0].to_string(sep=':', unit=units.hour),
                '15:50:10.484')
        self.assertEqual(radec[1].to_string(sep=':'), '1:53:00.3754')
