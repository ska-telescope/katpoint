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

"""Tests for the timestamp module."""

import pytest
import numpy as np
from astropy.time import Time

import katpoint


@pytest.mark.parametrize(
    'init_value, string',
    [
        (1248186982.3980861, '2009-07-21 14:36:22.398'),
        (Time('2009-07-21 02:52:12.34'), '2009-07-21 02:52:12.340'),
        (0, '1970-01-01 00:00:00'),
        (-10, '1969-12-31 23:59:50'),
        ('2009-07-21 02:52:12.034', '2009-07-21 02:52:12.034'),
        ('2009-07-21 02:52:12.000', '2009-07-21 02:52:12'),
        ('2009-07-21 02:52:12', '2009-07-21 02:52:12'),
        ('2009-07-21 02:52', '2009-07-21 02:52:00'),
        ('2009-07-21', '2009-07-21 00:00:00'),
        ('2009/07/21 02:52:12.034', '2009-07-21 02:52:12.034'),
        ('2009/07/21 02:52:12.000', '2009-07-21 02:52:12'),
        ('2009/07/21 02:52:12', '2009-07-21 02:52:12'),
        ('2009/07/21 02:52', '2009-07-21 02:52:00'),
        (b'2009/07/21', '2009-07-21 00:00:00'),
        (b'2020-07-17 12:40:12', '2020-07-17 12:40:12')
    ]
)
def test_construct_valid_timestamp(init_value, string):
    t = katpoint.Timestamp(init_value)
    assert str(t) == string, (
        "Timestamp string ('{}') differs from expected one ('{}')".format(str(t), string))
    # Exercise local() code path too
    print(t.local())


@pytest.mark.parametrize('init_value', ['gielie', '03 Mar 2003'])
def test_construct_invalid_timestamp(init_value):
    with pytest.raises(ValueError):
        katpoint.Timestamp(init_value)


def test_now_and_ordering_timestamps():
    t1 = Time.now()
    # We won't run this test during a leap second, promise...
    t2 = katpoint.Timestamp()
    assert t2 >= t1, "The second now() is not after the first now()"
    assert t2 - t1 >= 0.0
    t3 = katpoint.Timestamp()
    assert t2 <= t3, "The second now() is not before the third now()"


def test_numerical_timestamp():
    """Test numerical properties of timestamps."""
    t0 = 1248186982.3980861
    t = katpoint.Timestamp(t0)
    assert t == t + 0.0
    assert t != t + 1.0
    assert t > t - 1.0
    assert t < t + 1.0
    # This only works for scalars...
    assert t == eval('katpoint.' + repr(t))
    assert float(t) == t0

    t1 = Time('2009-07-21 02:52:12.34')
    t = katpoint.Timestamp(t1)
    t += 2.0
    t -= 2.0
    assert t.time == t1
    assert t / 2.0 == t * 0.5
    assert 1.0 + t == t + 1.0
    try:
        assert hash(t) == hash(t + 0.0), 'Timestamp hashes not equal'
    except TypeError:
        pytest.fail('Timestamp object not hashable')


def test_operators():
    """Test operators defined for timestamps."""
    t0 = 1248186982.3980861
    t = katpoint.Timestamp(t0)
    s = t.secs
    # Logical operators, float treated as absolute time
    assert t == s
    assert t < s + 1
    assert t > s - 1
    # Arithmetic operators, float treated as interval
    assert isinstance(t - s, katpoint.Timestamp)
    assert isinstance(s - t, float)
    assert isinstance(t - t, float)


def test_array_timestamps():
    t = katpoint.Timestamp([1234567890.0, 1234567891.0])
    with pytest.raises(TypeError):
        str(t)
    with pytest.raises(TypeError):
        float(t)
    with pytest.raises(TypeError):
        t.local()
    np.testing.assert_array_equal(t == 1234567890.0, [True, False])
    np.testing.assert_array_equal(t != 1234567890.0, [False, True])
    t2 = katpoint.Timestamp(1234567890.0)
    # Exercise various repr code paths
    print(repr(t2))
    print(repr(t2 + np.arange(0)))
    print(repr(t2 + np.arange(1)))
    print(repr(t2 + np.arange(2)))
    print(repr(t2 + np.arange(3)))


@pytest.mark.parametrize('mjd', [59000.0, (59000.0, 59001.0)])
def test_mjd_timestamp(mjd):
    t = Time(mjd, format='mjd')
    t2 = katpoint.Timestamp(t)
    assert np.all(t2.to_mjd() == mjd)
