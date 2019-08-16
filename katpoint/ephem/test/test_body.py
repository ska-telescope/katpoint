"""Tests for the body module."""

import unittest

from ephem import degrees
from ephem import hours
from ephem import FixedBody
from ephem import Mars
from ephem import Moon
from ephem import Sun
from ephem import Observer
from ephem import Date

class TestFixedBody(unittest.TestCase):
    """Test for the FixedBody class."""
    def test_compute(self):
        """Test compute method"""
        obs = Observer()
        obs.lat = degrees('10:00:00.000')
        obs.lon = degrees('80:00:00.000')
        obs.date = Date('2020/1/1')
        obs.pressure = 0.0

        ra = hours('10:10:40.123')
        dec = degrees('40:20:50.567')
        body = FixedBody()
        body._ra = ra
        body._dec = dec
        body.compute(obs)

        self.assertEqual(str(body.a_ra), '10:10:40.123')
        self.assertEqual(str(body.a_dec), '40:20:50.567')

        #self.assertEqual(str(body.ra), '10:11:51.77')
        #self.assertEqual(str(body.dec), '40:14:47.3')
        #self.assertEqual(str(body.ra), '10:11:51.2981')
        #self.assertEqual(str(body.dec), '40:14:47.181')

        #self.assertEqual(str(body.az), '326:05:54.8')
        self.assertEqual(str(body.az), '326:05:58.4284')
        #self.assertEqual(str(body.alt), '51:21:18.5')
        self.assertEqual(str(body.alt), '51:21:20.5645')

    def test_planet(self):
        obs = Observer()
        obs.lat = degrees('10:00:00.000')
        obs.lon = degrees('80:00:00.000')
        obs.date = Date('2020/1/1 00:00:00')
        obs.pressure = 0.0

        body = Mars()
        body.compute(obs)

        #self.assertEqual(str(body.a_ra), '15:43:47.22')
        #self.assertEqual(str(body.a_dec), '-19:23:07.0')

        #self.assertEqual(str(body.az), '118:10:06.1')
        #self.assertEqual(str(body.alt), '27:23:13.3')
        self.assertEqual(str(body.az), '118:10:04.6935')
        self.assertEqual(str(body.alt), '27:23:11.9665')

    def test_moon(self):
        obs = Observer()
        obs.lat = degrees('10:00:00.000')
        obs.lon = degrees('80:00:00.000')
        obs.date = Date('2020/1/1 10:00:00')
        obs.pressure = 0.0

        body = Moon()
        body.compute(obs)

        #self.assertEqual(str(body.a_ra), '15:43:47.22')
        #self.assertEqual(str(body.a_dec), '-19:23:07.0')

        #self.assertEqual(str(body.az), '127:15:23.6')
        #self.assertEqual(str(body.alt), '60:05:13.7')
        self.assertEqual(str(body.az), '127:15:45.1331')
        self.assertEqual(str(body.alt), '60:05:17.8175')

    def test_sun(self):
        obs = Observer()
        obs.lat = degrees('10:00:00.000')
        obs.lon = degrees('80:00:00.000')
        obs.date = Date('2020/1/1 10:00:00')
        obs.pressure = 0.0

        body = Sun()
        body.compute(obs)

        #self.assertEqual(str(body.a_ra), '15:43:47.22')
        #self.assertEqual(str(body.a_dec), '-19:23:07.0')

        #self.assertEqual(str(body.az), '234:53:20.8')
        #self.assertEqual(str(body.alt), '31:38:09.4')
        self.assertEqual(str(body.az), '234:53:18.9637')
        self.assertEqual(str(body.alt), '31:38:12.2448')
