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

import pytest
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle

import katpoint

from .helper import assert_angles_almost_equal


@pytest.mark.parametrize("angle, angle_deg", [('10:00:00', 10), ('10:45:00', 10.75), ('10.0', 10),
                                              ((10 * u.deg).to_value(u.rad), pytest.approx(10)),
                                              ('10d00m00s', 10), ((10, 0, 0), 10),
                                              ('10h00m00s', pytest.approx(150))])
def test_angle_from_degrees(angle, angle_deg):
    assert katpoint.conversion.to_angle(angle, sexagesimal_unit=u.deg).deg == angle_deg


@pytest.mark.parametrize("angle, angle_hour", [('10:00:00', 10), ('10:45:00', 10.75),
                                               ('150.0', pytest.approx(10)),
                                               ((150 * u.deg).to_value(u.rad), pytest.approx(10)),
                                               ('10h00m00s', 10), ((10, 0, 0), 10),
                                               ('10d00m00s', pytest.approx(10 / 15))])
def test_angle_from_hours(angle, angle_hour):
    assert katpoint.conversion.to_angle(angle, sexagesimal_unit=u.hour).hour == angle_hour


@pytest.fixture
def random_geoid():
    N = 1000
    lat = 0.999 * np.pi * (np.random.rand(N) - 0.5)
    lon = 2.0 * np.pi * np.random.rand(N)
    alt = 1000.0 * np.random.randn(N)
    return lat, lon, alt


def test_lla_to_ecef(random_geoid):
    """Closure tests for LLA to ECEF conversion and vice versa."""
    lat, lon, alt = random_geoid
    x, y, z = katpoint.lla_to_ecef(lat, lon, alt)
    new_lat, new_lon, new_alt = katpoint.ecef_to_lla(x, y, z)
    new_x, new_y, new_z = katpoint.lla_to_ecef(new_lat, new_lon, new_alt)
    assert_angles_almost_equal(new_lat, lat, decimal=12)
    assert_angles_almost_equal(new_lon, lon, decimal=12)
    assert_angles_almost_equal(new_alt, alt, decimal=6)
    np.testing.assert_almost_equal(new_x, x, decimal=8)
    np.testing.assert_almost_equal(new_y, y, decimal=8)
    np.testing.assert_almost_equal(new_z, z, decimal=6)
    # Test alternate version of ecef_to_lla2
    new_lat2, new_lon2, new_alt2 = katpoint.conversion.ecef_to_lla2(x, y, z)
    assert_angles_almost_equal(new_lat2, lat, decimal=12)
    assert_angles_almost_equal(new_lon2, lon, decimal=12)
    assert_angles_almost_equal(new_alt2, alt, decimal=6)


def test_ecef_to_enu(random_geoid):
    """Closure tests for ECEF to ENU conversion and vice versa."""
    lat, lon, alt = random_geoid
    x, y, z = katpoint.lla_to_ecef(lat, lon, alt)
    e, n, u = katpoint.ecef_to_enu(lat[0], lon[0], alt[0], x, y, z)
    new_x, new_y, new_z = katpoint.enu_to_ecef(lat[0], lon[0], alt[0], e, n, u)
    np.testing.assert_almost_equal(new_x, x, decimal=8)
    np.testing.assert_almost_equal(new_y, y, decimal=8)
    np.testing.assert_almost_equal(new_z, z, decimal=8)


@pytest.fixture
def random_sphere():
    N = 1000
    az = Angle(2.0 * np.pi * np.random.rand(N), unit=u.rad)
    el = Angle(0.999 * np.pi * (np.random.rand(N) - 0.5), unit=u.rad)
    return az, el


def test_azel_to_enu(random_sphere):
    """Closure tests for (az, el) to ENU conversion and vice versa."""
    az, el = random_sphere
    e, n, u = katpoint.azel_to_enu(az.rad, el.rad)
    new_az, new_el = katpoint.enu_to_azel(e, n, u)
    assert_angles_almost_equal(new_az, az.rad, decimal=15)
    assert_angles_almost_equal(new_el, el.rad, decimal=15)
