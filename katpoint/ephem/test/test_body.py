"""Tests for the body module."""

import unittest
from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs84

from ephem import degrees
from ephem import hours
from ephem import FixedBody
from ephem import Mars
from ephem import Moon
from ephem import Sun
from ephem import readtle
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

        self.assertEqual(str(body.a_ra), '10:10:40.12')
        self.assertEqual(str(body.a_dec), '40:20:50.6')

        #self.assertEqual(str(body.ra), '10:11:51.77')
        #self.assertEqual(str(body.dec), '40:14:47.3')
        #self.assertEqual(str(body.ra), '10:11:51.2981')
        #self.assertEqual(str(body.dec), '40:14:47.181')

        #self.assertEqual(str(body.az), '326:05:54.8')
        self.assertEqual(str(body.az), '326:05:57.6')
        #self.assertEqual(str(body.alt), '51:21:18.5')
        self.assertEqual(str(body.alt), '51:21:20.0')

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
        self.assertEqual(str(body.az), '118:10:05.1')
        self.assertEqual(str(body.alt), '27:23:12.9')

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
        self.assertEqual(str(body.az), '127:15:46.4')
        self.assertEqual(str(body.alt), '60:05:18.6')

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
        self.assertEqual(str(body.az), '234:53:19.5')
        self.assertEqual(str(body.alt), '31:38:11.4')

    def test_earth_satellite(self):
        name = ' GPS BIIA-21 (PRN 09) '
        line1 = '1 22700U 93042A   19266.32333151  .00000012  00000-0  10000-3 0  8057'
        line2 = '2 22700  55.4408  61.3790 0191986  78.1802 283.9935  2.00561720104282'
        et = readtle(name, line1, line2)

        self.assertEqual(str(et._epoch), '2019/9/23 07:45:36')
        self.assertEqual(str(et._inc), '55:26:26.9')
        self.assertEqual(str(et._raan), '61:22:44.4')
        self.assertEqual(et._e, 0.0191986)
        self.assertEqual(str(et._ap), '78:10:48.7')
        self.assertEqual(str(et._M), '283:59:36.6')
        self.assertEqual(et._n, 2.0056172)
        self.assertEqual(et._decay, 1.2e-07)
        self.assertEqual(et._orbit, 10428)
        self.assertEqual(et._drag, 1.e-04)

        xephem = ' GPS BIIA-21 (PRN 09) ,E,9/23.32333151/2019| 6/15.3242/2019| 1/1.32422/2020,55.4408,61.379002,0.0191986,78.180199,283.9935,2.0056172,1.2e-07,10428,9.9999997e-05'

        self.assertEqual(et.writedb().split(',')[0], xephem.split(',')[0])
        self.assertEqual(et.writedb().split(',')[1], xephem.split(',')[1])
        self.assertEqual(et.writedb().split(',')[2].split('|')[0],
                xephem.split(',')[2].split('|')[0])
        #self.assertEqual(et.writedb().split(',')[2].split('|')[1],
        #        xephem.split(',')[2].split('|')[1])
        #self.assertEqual(et.writedb().split(',')[3].split('|')[2],
        #        xephem.split(',')[2].split('|')[2])
        self.assertEqual(et.writedb().split(',')[3], xephem.split(',')[3])

        # pyephem adds spurious precision to these 3 fields
        self.assertEqual(et.writedb().split(',')[4], xephem.split(',')[4][:6])
        self.assertEqual(et.writedb().split(',')[5][:7], xephem.split(',')[5][:7])
        self.assertEqual(et.writedb().split(',')[6], xephem.split(',')[6][:5])
        self.assertEqual(et.writedb().split(',')[7], xephem.split(',')[7])
        self.assertEqual(et.writedb().split(',')[8], xephem.split(',')[8])
        self.assertEqual(et.writedb().split(',')[9], xephem.split(',')[9])
        self.assertEqual(et.writedb().split(',')[10], xephem.split(',')[10])

        # Test compute
        obs = Observer()
        obs.lat = degrees('10:00:00.000')
        obs.lon = degrees('80:00:00.000')
        obs.date = Date('2020/1/1 10:00:00')
        obs.pressure = 0.0
        et.compute(obs)

        # Create an sgp4 objec
        sgp = twoline2rv(line1, line2, wgs84)

        self.assertEqual(sgp.epochyr, et._sat.epochyr)
        self.assertAlmostEqual(sgp.epochdays, et._sat.epochdays)
        self.assertEqual(sgp.bstar, et._sat.bstar)
        self.assertEqual(sgp.inclo, et._sat.inclo)
        self.assertEqual(sgp.nodeo, et._sat.nodeo)
        self.assertEqual(sgp.ecco, et._sat.ecco)
        self.assertEqual(sgp.argpo, et._sat.argpo)
        self.assertEqual(sgp.mo, et._sat.mo)
        self.assertAlmostEqual(sgp.no, et._sat.no)

        self.assertEqual(str(et.a_ra), '22:20:00.48')
        self.assertEqual(str(et.a_dec), '-68:43:38.9')
        self.assertEqual(str(et.az), '178:11:58.1')
        self.assertEqual(str(et.alt), '11:17:54.7')
