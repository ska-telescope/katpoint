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

"""Tests for the antenna module."""

import time
import pickle

import pytest
import numpy as np

import katpoint

from .helper import assert_angles_almost_equal


@pytest.mark.parametrize(
    "description",
    [
        'XDM, -25:53:23.0, 27:41:03.0, 1406.1086, 15.0',
        'FF1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 18.4 -8.7 0.0',
        ('FF2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 86.2 25.5 0.0, '
         '-0:06:39.6 0 0 0 0 0 0:09:48.9, 1.16'),
    ]
)
def test_construct_valid_antenna(description):
    """Test construction of valid antennas from strings and vice versa."""
    # Normalise description string through one cycle to allow comparison
    reference_description = katpoint.Antenna(description).description
    test_antenna = katpoint.Antenna(reference_description)
    assert test_antenna.description == reference_description, (
        'Antenna description differs from original string')
    assert test_antenna.description == test_antenna.format_katcp(), (
        'Antenna description differs from KATCP format')
    # Exercise repr() and str()
    print('{!r} {}'.format(test_antenna, test_antenna))


@pytest.mark.parametrize("description", ['XDM, -25:53:23.05075, 27:41:03.0', ''])
def test_construct_invalid_antenna(description):
    """Test construction of invalid antennas from strings."""
    with pytest.raises(ValueError):
        katpoint.Antenna(description)


def test_construct_antenna():
    """Test construction of antennas from strings and vice versa."""
    descr = katpoint.Antenna('XDM, -25:53:23.0, 27:41:03.0, 1406.1086, 15.0').description
    assert descr == katpoint.Antenna(*descr.split(', ')).description
    with pytest.raises(ValueError):
        katpoint.Antenna(descr, *descr.split(', ')[1:])
    # Check that description string updates when object is updated
    a1 = katpoint.Antenna('FF1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 18.4 -8.7 0.0')
    a2 = katpoint.Antenna('FF2, -30:43:17.3, 21:24:38.5, 1038.0, 13.0, 18.4 -8.7 0.0, 0.1, 1.22')
    assert a1 != a2, 'Antennas should be inequal'
    a1.name = 'FF2'
    a1.diameter = 13.0
    a1.pointing_model = katpoint.PointingModel('0.1')
    a1.beamwidth = 1.22
    assert a1.description == a2.description, 'Antenna description string not updated'
    assert a1 == a2.description, 'Antenna not equal to description string'
    assert a1 == a2, 'Antennas not equal'
    assert a1 == katpoint.Antenna(a2), 'Construction with antenna object failed'
    assert a1 == pickle.loads(pickle.dumps(a1)), 'Pickling failed'
    try:
        assert hash(a1) == hash(a2), 'Antenna hashes not equal'
    except TypeError:
        pytest.fail('Antenna object not hashable')


def test_local_sidereal_time():
    """Test sidereal time and the use of date/time strings vs floats as timestamps."""
    ant = katpoint.Antenna('XDM, -25:53:23.0, 27:41:03.0, 1406.1086, 15.0')
    timestamp = '2009/07/07 08:36:20'
    utc_secs = time.mktime(time.strptime(timestamp, '%Y/%m/%d %H:%M:%S')) - time.timezone
    sid1 = ant.local_sidereal_time(timestamp)
    sid2 = ant.local_sidereal_time(utc_secs)
    assert sid1 == sid2, 'Sidereal time differs for float and date/time string'
    sid3 = ant.local_sidereal_time([timestamp, timestamp])
    sid4 = ant.local_sidereal_time([utc_secs, utc_secs])
    assert_angles_almost_equal(np.array([a.rad for a in sid3]),
                               np.array([a.rad for a in sid4]), decimal=12)


def test_array_reference_antenna():
    ant = katpoint.Antenna('FF2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 86.2 25.5 0.0, '
                           '-0:06:39.6 0 0 0 0 0 0:09:48.9, 1.16')
    ref_ant = ant.array_reference_antenna()
    assert ref_ant.description == 'array, -30:43:17.3, 21:24:38.5, 1038, 12.0, , , 1.16'
