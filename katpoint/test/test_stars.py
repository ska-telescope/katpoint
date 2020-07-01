"""Tests for the stars module."""

import unittest
import numpy as np

from katpoint.stars import readdb


class test_stars(unittest.TestCase):
    def test_earth_satellite(self):

        record = 'GPS BIIA-21 (PR,E,9/23.32333151/2019| 6/15.3242/2019| 1/1.32422/2020,' \
                 '55.4408,61.379002,0.0191986,78.180199,283.9935,2.0056172,1.2e-07,10428,9.9999997e-05'

        e = readdb(record)
        self.assertEqual(e.name, 'GPS BIIA-21 (PR')
        self.assertEqual(str(e._epoch), '2019-09-23 07:45:35.842')
        self.assertEqual(e._inc, np.deg2rad(55.4408))
        self.assertEqual(e._raan, np.deg2rad(61.379002))
        self.assertEqual(e._e, 0.0191986)
        self.assertEqual(e._ap, np.deg2rad(78.180199))
        self.assertEqual(e._M, np.deg2rad(283.9935))
        self.assertEqual(e._n, 2.0056172)
        self.assertEqual(e._decay, 1.2e-07)
        self.assertEqual(e._orbit, 10428)
        self.assertEqual(e._drag, 9.9999997e-05)

    def test_star(self):
        record = 'Sadr,f|S|F8,20:22:13.7|2.43,40:15:24|-0.93,2.23,2000,0'
        readdb(record)
