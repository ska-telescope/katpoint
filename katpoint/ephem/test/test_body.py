"""Tests for the body module."""

import unittest

from ephem import degrees
from ephem import hours
from ephem import FixedBody
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
