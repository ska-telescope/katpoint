"""Tests for the observer module."""

import unittest

from ephem import degrees
from ephem import hours
from ephem import FixedBody
from ephem import Observer
from ephem import Date

class TestObserver(unittest.TestCase):
    """Test for the Observer class."""
    def test_radec_of(self):
        """Test radec_of method"""
        obs = Observer()
        obs.lat = degrees('10:00:00.000')
        obs.lon = degrees('80:00:00.000')
        obs.date = Date('2020/1/1')
        obs.pressure = 0.0

        az = degrees('10:10:40.123')
        alt = degrees('40:20:50.567')
        radec = obs.radec_of(az, alt)

        #self.assertEqual(str(radec[0]), '12:59:07.12')
        #self.assertEqual(str(radec[1]), '58:26:58.6')
        self.assertEqual(str(radec[0]), '12:59:06.23')
        self.assertEqual(str(radec[1]), '58:26:47.1')

    def test_sidereal_time(self):
        """Test sidereal_time method"""
        obs = Observer()
        obs.lat = degrees('10:00:00.000')
        obs.lon = degrees('80:00:00.000')
        obs.date = Date('2020/1/1 10:00:00')

        st = obs.sidereal_time()

        #self.assertEqual(str(st), '22:02:06.79')
        self.assertEqual(str(st), '22:02:06.61')
