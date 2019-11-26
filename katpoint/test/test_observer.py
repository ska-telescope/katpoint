"""Tests for the observer module.

The values in the comments are those produced by the real pyephem
"""

import unittest
from astropy.time import Time
from astropy import coordinates
from astropy import units

import katpoint

class TestObserver(unittest.TestCase):
    """Test for the Observer class."""
    def test_radec_of(self):
        """Test radec_of method"""
        obs = katpoint.Observer()
        obs.lat = coordinates.Latitude('10:00:00.000', unit=units.deg)
        obs.lon = coordinates.Longitude('80:00:00.000', unit=units.deg)
        obs.date = Time('2020-01-01 00:00:00.000')
        obs.pressure = 0.0

        az = coordinates.Longitude('10:10:40.123', unit=units.deg)
        alt = coordinates.Latitude('40:20:50.567', unit=units.deg)
        radec = obs.radec_of(az, alt)

        # 12:59:07.12 58:26:58.6
        self.assertEqual(radec[0].to_string(sep=':', unit=units.hour),
                '12:59:06.9264')
        self.assertEqual(radec[1].to_string(sep=':'), '58:26:58.6368')

    def test_sidereal_time(self):
        """Test sidereal_time method"""
        obs = katpoint.Observer()
        obs.lat = coordinates.Latitude('10:00:00.000', unit=units.deg)
        obs.lon = coordinates.Longitude('80:00:00.000', unit=units.deg)
        obs.date = Time('2020-01-01 10:00:00.000')

        st = obs.sidereal_time()

        # 22:02:06.79
        self.assertEqual(st.to_string(sep=':', unit=units.hour), '22:02:06.6175')
