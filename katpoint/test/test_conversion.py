################################################################################
# Copyright (c) 2009-2020, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Tests for the conversion module."""

import unittest

import numpy as np
import astropy.units as u
from astropy.coordinates import Angle

import katpoint


def assert_angles_almost_equal(x, y, decimal):
    def primary_angle(x):
        return x - np.round(x / (2.0 * np.pi)) * 2.0 * np.pi
    np.testing.assert_almost_equal(primary_angle(x - y), np.zeros(np.shape(x)), decimal=decimal)


class TestGeodetic(unittest.TestCase):
    """Closure tests for geodetic coordinate transformations."""

    def setUp(self):
        N = 1000
        self.lat = 0.999 * np.pi * (np.random.rand(N) - 0.5)
        self.lon = 2.0 * np.pi * np.random.rand(N)
        self.alt = 1000.0 * np.random.randn(N)

    def test_lla_to_ecef(self):
        """Closure tests for LLA to ECEF conversion and vice versa."""
        x, y, z = katpoint.lla_to_ecef(self.lat, self.lon, self.alt)
        new_lat, new_lon, new_alt = katpoint.ecef_to_lla(x, y, z)
        new_x, new_y, new_z = katpoint.lla_to_ecef(new_lat, new_lon, new_alt)
        assert_angles_almost_equal(new_lat, self.lat, decimal=12)
        assert_angles_almost_equal(new_lon, self.lon, decimal=12)
        assert_angles_almost_equal(new_alt, self.alt, decimal=6)
        np.testing.assert_almost_equal(new_x, x, decimal=8)
        np.testing.assert_almost_equal(new_y, y, decimal=8)
        np.testing.assert_almost_equal(new_z, z, decimal=6)
        if hasattr(katpoint, '_conversion'):
            new_lat2, new_lon2, new_alt2 = katpoint._conversion.ecef_to_lla2(x, y, z)
            assert_angles_almost_equal(new_lat2, self.lat, decimal=12)
            assert_angles_almost_equal(new_lon2, self.lon, decimal=12)
            assert_angles_almost_equal(new_alt2, self.alt, decimal=6)

    def test_ecef_to_enu(self):
        """Closure tests for ECEF to ENU conversion and vice versa."""
        x, y, z = katpoint.lla_to_ecef(self.lat, self.lon, self.alt)
        e, n, u = katpoint.ecef_to_enu(self.lat[0], self.lon[0], self.alt[0], x, y, z)
        new_x, new_y, new_z = katpoint.enu_to_ecef(self.lat[0], self.lon[0], self.alt[0], e, n, u)
        np.testing.assert_almost_equal(new_x, x, decimal=8)
        np.testing.assert_almost_equal(new_y, y, decimal=8)
        np.testing.assert_almost_equal(new_z, z, decimal=8)


class TestSpherical(unittest.TestCase):
    """Closure tests for spherical coordinate transformations."""

    def setUp(self):
        N = 1000
        self.az = Angle(2.0 * np.pi * np.random.rand(N), unit=u.rad)
        self.el = Angle(0.999 * np.pi * (np.random.rand(N) - 0.5), unit=u.rad)

    def test_azel_to_enu(self):
        """Closure tests for (az, el) to ENU conversion and vice versa."""
        e, n, u = katpoint.azel_to_enu(self.az.rad, self.el.rad)
        new_az, new_el = katpoint.enu_to_azel(e, n, u)
        assert_angles_almost_equal(new_az, self.az.rad, decimal=15)
        assert_angles_almost_equal(new_el, self.el.rad, decimal=15)
