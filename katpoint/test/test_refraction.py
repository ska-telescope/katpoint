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

import numpy as np
import pytest

import katpoint

from .helper import assert_angles_almost_equal


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
