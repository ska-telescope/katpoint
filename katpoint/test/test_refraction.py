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

"""Tests for the refraction module."""

import pytest
import numpy as np
import astropy.constants as const
from astropy.coordinates import EarthLocation

import katpoint
from katpoint.refraction import SaastamoinenZenithDelay
from katpoint.test.helper import assert_angles_almost_equal

try:
    from almacalc.lowlevel import sastd, sastw, gmf11
except ImportError:
    HAS_ALMACALC = False
else:
    HAS_ALMACALC = True


def test_refraction_basic():
    """Test basic refraction correction properties."""
    rc = katpoint.RefractionCorrection()
    print(repr(rc))
    with pytest.raises(ValueError):
        katpoint.RefractionCorrection('unknown')
    rc2 = katpoint.RefractionCorrection()
    assert rc == rc2, 'Refraction models should be equal'
    try:
        assert hash(rc) == hash(rc2), 'Refraction model hashes should be equal'
    except TypeError:
        pytest.fail('RefractionCorrection object not hashable')


def test_refraction_closure():
    """Test closure between refraction correction and its reverse operation."""
    rc = katpoint.RefractionCorrection()
    el = np.radians(np.arange(0.0, 90.1, 0.1))
    # Generate random meteorological data (a single measurement, hopefully sensible)
    temp = -10. + 50. * np.random.rand()
    pressure = 900. + 200. * np.random.rand()
    humidity = 5. + 90. * np.random.rand()
    # Test closure on el grid
    refracted_el = rc.apply(el, temp, pressure, humidity)
    reversed_el = rc.reverse(refracted_el, temp, pressure, humidity)
    assert_angles_almost_equal(reversed_el, el, decimal=7,
                               err_msg='Elevation closure error for temp=%f, pressure=%f, humidity=%f' %
                                       (temp, pressure, humidity))
    # Generate random meteorological data, now one weather measurement per elevation value
    temp = -10. + 50. * np.random.rand(len(el))
    pressure = 900. + 200. * np.random.rand(len(el))
    humidity = 5. + 90. * np.random.rand(len(el))
    # Test closure on el grid
    refracted_el = rc.apply(el, temp, pressure, humidity)
    reversed_el = rc.reverse(refracted_el, temp, pressure, humidity)
    assert_angles_almost_equal(reversed_el, el, decimal=7,
                               err_msg='Elevation closure error for temp=%s, pressure=%s, humidity=%s' %
                                       (temp, pressure, humidity))


@pytest.mark.skipif(not HAS_ALMACALC, reason="almacalc is not installed")
@pytest.mark.parametrize(
    "latitude,longitude,height",
    [
        ('-25:53:23.0', '27:41:03.0', 1406.0),
        ('35:00:00.0', '-40:00:00.0', 0.0),
        ('85:00:00.0', '170:00:00.0', -200.0),
        ('-35:00:00.0', '-40:00:00.0', 6000.0),
        ('-90:00:00.0', '180:00:00.0', 200.0),
    ]
)
def test_zenith_delay(latitude, longitude, height):
    """Test hydrostatic and wet zenith delays against AlmaCalc."""
    location = EarthLocation.from_geodetic(longitude, latitude, height)
    zd = SaastamoinenZenithDelay(location)
    pressure = np.arange(800., 1000., 5.)
    actual = zd.hydrostatic(0, pressure, 0) * const.c
    expected = sastd(pressure, location.lat.rad, location.height.value)
    np.testing.assert_allclose(actual.value, expected, atol=1e-14)
    for temperature in np.arange(-5., 45., 5.):
        for humidity in np.arange(0., 1.05, 0.05):
            actual = zd.wet(temperature, 0, humidity) * const.c
            expected = sastw(humidity, temperature)
            assert actual.value == pytest.approx(expected, abs=1e-14)
