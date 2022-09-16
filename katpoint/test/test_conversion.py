################################################################################
# Copyright (c) 2009-2022, National Research Foundation (SARAO)
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

# pylint: disable=missing-function-docstring

import pytest
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import Angle

import katpoint

from .helper import assert_angles_almost_equal


@pytest.mark.parametrize("angle, angle_deg", [('10:00:00', 10),
                                              ('10:45:00', 10.75),
                                              ('10.0', 10),
                                              ((10 * u.deg).to_value(u.rad), pytest.approx(10)),
                                              ('10d00m00s', 10),
                                              ('10h00m00s', pytest.approx(150))])
def test_angle_from_degrees(angle, angle_deg):
    assert katpoint.conversion.to_angle(angle, sexagesimal_unit=u.deg).deg == angle_deg


@pytest.mark.parametrize("angle, angle_hour", [('10:00:00', 10),
                                               ('10:45:00', 10.75),
                                               ('150.0', pytest.approx(10)),
                                               ((150 * u.deg).to_value(u.rad), pytest.approx(10)),
                                               ('10h00m00s', 10),
                                               ('10d00m00s', pytest.approx(10 / 15))])
def test_angle_from_hours(angle, angle_hour):
    assert katpoint.conversion.to_angle(angle, sexagesimal_unit=u.hour).hour == angle_hour


def test_bytes_to_angle():
    # Raw bytes are not supported
    with pytest.raises(TypeError):
        katpoint.conversion.to_angle(b'1.2')
    # You probably meant this:
    assert katpoint.conversion.to_angle(b'1.2'.decode()) == 1.2 * u.deg
    # But some strange folks might intend this instead:
    np.testing.assert_array_equal(katpoint.conversion.to_angle(Angle(b'1.2', unit=u.deg)),
                                  np.array([49, 46, 50]) * u.deg)


@pytest.mark.parametrize(
    "angle, kwargs, angle_string",
    [
        # Basic zero stripping like %g
        ('10:20:30.4d', dict(), '10:20:30.4d'),
        ('10:20:30.00d', dict(), '10:20:30d'),
        ('10:20:30.4000d', dict(), '10:20:30.4d'),
        # Hour angle checks
        ('10:20:30.4h', dict(), '10:20:30.4h'),
        ('10d', dict(unit=u.hour, pad=True), '00:40:00h'),
        # Precision checks
        ('10:20:30.12d', dict(precision=4), '10:20:30.12d'),
        ('10:20:30.123487d', dict(precision=4), '10:20:30.1235d'),
        (10.123487 * u.deg, dict(decimal=True, precision=4), '10.1235d'),
        # Maximum precision
        (10.1234567890123 * u.deg, dict(), '10:07:24.44444044d'),
        (10.1234567890123 * u.deg, dict(decimal=True), '10.123456789012d'),
        (10.12345678901234 * u.hour, dict(), '10:07:24.444440444h'),
        (10.12345678901234 * u.hour, dict(decimal=True), '10.1234567890123h'),
        # Multidimensional example
        ([['10d', '20d']], dict(decimal=True), [['10d', '20d']]),
        # Convert unit and tolerate non-standard separator if decimal
        (np.pi * u.rad, dict(unit=u.deg, sep='dms', decimal=True), '180d'),
        # Don't display unit
        ('12d34m56s', dict(show_unit=False), '12:34:56'),
        (10.123 * u.deg, dict(decimal=True, show_unit=False), '10.123'),
        (10.123 * u.hour, dict(decimal=True, show_unit=False), '10.123'),
    ]
)
def test_angle_to_string(angle, kwargs, angle_string):
    np.testing.assert_array_equal(katpoint.conversion.angle_to_string(Angle(angle), **kwargs),
                                  angle_string)


@pytest.mark.parametrize("angle, kwargs", [('10:20:30.4d', dict(unit=u.rad)),
                                           ('10rad', dict()),
                                           ('10d', dict(sep='dms'))])
def test_angle_to_string_errors(angle, kwargs):
    with pytest.raises(ValueError):
        katpoint.conversion.angle_to_string(Angle(angle), **kwargs)


@pytest.mark.parametrize("kwargs", [dict(),
                                    dict(decimal=True),
                                    dict(unit=u.hour),
                                    dict(unit=u.hour, decimal=True)])
def test_angle_to_string_round_trip(kwargs):
    rs = np.random.RandomState(46)  # pylint: disable=no-member
    angle1 = Angle(360 * rs.rand(1000) * u.deg)
    string1 = katpoint.conversion.angle_to_string(angle1, **kwargs)
    angle2 = katpoint.conversion.to_angle(string1)
    # The smallest angle we care about is a micron held at a distance of the Earth's diameter
    atol = (1 * u.micron) / (2 * const.R_earth) * u.rad
    assert np.allclose(angle2, angle1, rtol=0, atol=atol)
    string2 = katpoint.conversion.angle_to_string(angle2, **kwargs)
    # The strings round-trip exactly, which is good because most Angles start life as strings
    np.testing.assert_array_equal(string2, string1)


@pytest.fixture(name='random_geoid')
def fixture_random_geoid():
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
    east, north, up = katpoint.ecef_to_enu(lat[0], lon[0], alt[0], x, y, z)
    new_x, new_y, new_z = katpoint.enu_to_ecef(lat[0], lon[0], alt[0], east, north, up)
    np.testing.assert_almost_equal(new_x, x, decimal=8)
    np.testing.assert_almost_equal(new_y, y, decimal=8)
    np.testing.assert_almost_equal(new_z, z, decimal=8)


@pytest.fixture(name='random_sphere')
def fixture_random_sphere():
    N = 1000
    az = Angle(2.0 * np.pi * np.random.rand(N), unit=u.rad)
    el = Angle(0.999 * np.pi * (np.random.rand(N) - 0.5), unit=u.rad)
    return az, el


def test_azel_to_enu(random_sphere):
    """Closure tests for (az, el) to ENU conversion and vice versa."""
    az, el = random_sphere
    east, north, up = katpoint.azel_to_enu(az.rad, el.rad)
    new_az, new_el = katpoint.enu_to_azel(east, north, up)
    assert_angles_almost_equal(new_az, az.rad, decimal=15)
    assert_angles_almost_equal(new_el, el.rad, decimal=15)
