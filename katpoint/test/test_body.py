"""Tests for the body module.

The values in the comments are the results from the same tests run on the
pyephem package.

"""

import unittest
import numpy as np
from astropy.time import Time
from astropy import coordinates
from astropy import units
from katpoint import Observer

from katpoint.bodies import FixedBody
from katpoint.bodies import Mars
from katpoint.bodies import Moon
from katpoint.bodies import Sun
from katpoint.bodies import readtle

class TestFixedBody(unittest.TestCase):
    """Test for the FixedBody class."""
    def test_compute(self):
        """Test compute method"""
        obs = Observer()
        obs.lat = coordinates.Latitude('10:00:00.000', unit=units.deg)
        obs.lon = coordinates.Longitude('80:00:00.000', unit=units.deg)
        obs.date = Time('2020-01-01 00:00:00.000')
        obs.pressure = 0.0

        ra = coordinates.Longitude('10:10:40.123', unit=units.hour)
        dec = coordinates.Latitude('40:20:50.567', unit=units.deg)
        body = FixedBody()
        body._ra = ra
        body._dec = dec
        body.compute(obs)

        self.assertEqual(body.a_ra.to_string(sep=':', unit=units.hour),
                '10:10:40.123')
        self.assertEqual(body.a_dec.to_string(sep=':'), '40:20:50.567')

        # 326:05:54.8 51:21:18.5
        self.assertEqual(body.az.to_string(sep=':'), '326:05:57.5415')
        self.assertEqual(body.alt.to_string(sep=':'), '51:21:20.0121')

    def test_planet(self):
        obs = Observer()
        obs.lat = coordinates.Latitude('10:00:00.000', unit=units.deg)
        obs.lon = coordinates.Longitude('80:00:00.000', unit=units.deg)
        obs.date = Time('2020-01-01 00:00:00.000')
        obs.pressure = 0.0

        body = Mars()
        body.compute(obs)

        # '118:10:06.1' '27:23:13.3'
        self.assertEqual(body.az.to_string(sep=':'), '118:10:05.1129')
        self.assertEqual(body.alt.to_string(sep=':'), '27:23:12.8494')

    def test_moon(self):
        obs = Observer()
        obs.lat = coordinates.Latitude('10:00:00.000', unit=units.deg)
        obs.lon = coordinates.Longitude('80:00:00.000', unit=units.deg)
        obs.date = Time('2020-01-01 10:00:00.000')
        obs.pressure = 0.0

        body = Moon()
        body.compute(obs)

        # 127:15:23.6 60:05:13.7'
        self.assertEqual(body.az.to_string(sep=':'), '127:15:17.1374')
        self.assertEqual(body.alt.to_string(sep=':'), '60:05:10.2433')

    def test_sun(self):
        obs = Observer()
        obs.lat = coordinates.Latitude('10:00:00.000', unit=units.deg)
        obs.lon = coordinates.Longitude('80:00:00.000', unit=units.deg)
        obs.date = Time('2020-01-01 10:00:00.000')
        obs.pressure = 0.0

        body = Sun()
        body.compute(obs)

        # 234:53:20.8 '31:38:09.4'
        self.assertEqual(body.az.to_string(sep=':'), '234:53:19.4833')
        self.assertEqual(body.alt.to_string(sep=':'), '31:38:11.4125')

    def test_earth_satellite(self):
        name = ' GPS BIIA-21 (PRN 09) '
        line1 = '1 22700U 93042A   19266.32333151  .00000012  00000-0  10000-3 0  8057'
        line2 = '2 22700  55.4408  61.3790 0191986  78.1802 283.9935  2.00561720104282'
        sat = readtle(name, line1, line2)

        # Check that the EarthSatellite object has the expect attribute
        # values.
        self.assertEqual(str(sat._epoch), '2019-09-23 07:45:35.842')
        self.assertEqual(sat._inc, np.deg2rad(55.4408))
        self.assertEqual(sat._raan, np.deg2rad(61.3790))
        self.assertEqual(sat._e, 0.0191986)
        self.assertEqual(sat._ap, np.deg2rad(78.1802))
        self.assertEqual(sat._M, np.deg2rad(283.9935))
        self.assertEqual(sat._n, 2.0056172)
        self.assertEqual(sat._decay, 1.2e-07)
        self.assertEqual(sat._orbit, 10428)
        self.assertEqual(sat._drag, 1.e-04)

        # This is xephem database record that pyephem generates
        xephem = ' GPS BIIA-21 (PRN 09) ,E,9/23.32333151/2019| 6/15.3242/2019| 1/1.32422/2020,55.4408,61.379002,0.0191986,78.180199,283.9935,2.0056172,1.2e-07,10428,9.9999997e-05'

        rec = sat.writedb()
        self.assertEqual(rec.split(',')[0], xephem.split(',')[0])
        self.assertEqual(rec.split(',')[1], xephem.split(',')[1])

        self.assertEqual(rec.split(',')[2].split('|')[0].split('/')[0],
                xephem.split(',')[2].split('|')[0].split('/')[0])
        self.assertAlmostEqual(float(rec.split(',')[2].split('|')[0].split('/')[1]),
                float(xephem.split(',')[2].split('|')[0].split('/')[1]))
        self.assertEqual(rec.split(',')[2].split('|')[0].split('/')[2],
                xephem.split(',')[2].split('|')[0].split('/')[2])

        self.assertEqual(rec.split(',')[2].split('|')[1].split('/')[0],
                xephem.split(',')[2].split('|')[1].split('/')[0])
        self.assertAlmostEqual(float(rec.split(',')[2].split('|')[1].split('/')[1]),
                float(xephem.split(',')[2].split('|')[1].split('/')[1]), places=2)
        self.assertEqual(rec.split(',')[2].split('|')[1].split('/')[2],
                xephem.split(',')[2].split('|')[1].split('/')[2])

        self.assertEqual(rec.split(',')[2].split('|')[2].split('/')[0],
                xephem.split(',')[2].split('|')[2].split('/')[0])
        self.assertAlmostEqual(float(rec.split(',')[2].split('|')[2].split('/')[1]),
                float(xephem.split(',')[2].split('|')[2].split('/')[1]), places=2)
        self.assertEqual(rec.split(',')[2].split('|')[2].split('/')[2],
                xephem.split(',')[2].split('|')[2].split('/')[2])

        self.assertEqual(rec.split(',')[3], xephem.split(',')[3])

        # pyephem adds spurious precision to these 3 fields
        self.assertEqual(rec.split(',')[4], xephem.split(',')[4][:6])
        self.assertEqual(rec.split(',')[5][:7], xephem.split(',')[5][:7])
        self.assertEqual(rec.split(',')[6], xephem.split(',')[6][:5])

        self.assertEqual(rec.split(',')[7], xephem.split(',')[7])
        self.assertEqual(rec.split(',')[8], xephem.split(',')[8])
        self.assertEqual(rec.split(',')[9], xephem.split(',')[9])
        self.assertEqual(rec.split(',')[10], xephem.split(',')[10])

        # Test compute
        obs = Observer()
        obs.lat = coordinates.Latitude('10:00:00.000', unit=units.deg)
        obs.lon = coordinates.Longitude('80:00:00.000', unit=units.deg)
        obs.date = Time('2019-09-23 07:45:36.000')
        obs.elevation = 4200.0
        obs.pressure = 0.0
        sat.compute(obs)

        # 3:32:59.21' '-2:04:36.3'
        self.assertEqual(sat.a_ra.to_string(sep=':', unit=units.hour),
                '3:32:56.7813')
        self.assertEqual(sat.a_dec.to_string(sep=':'), '-2:04:35.4329')

        # 280:32:07.2 -54:06:14.4
        self.assertEqual(sat.az.to_string(sep=':'), '280:32:29.675')
        self.assertEqual(sat.alt.to_string(sep=':'), '-54:06:50.7456')
