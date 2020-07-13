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
        ('2009/07/21', '2009-07-21 00:00:00'),
        ('2019-07-21 02:52:12', '2019-07-21 02:52:12')
    ]
)
def test_construct_valid_timestamp(init_value, string):
    t = katpoint.Timestamp(init_value)
    assert str(t) == string, (
        "Timestamp string ('{}') differs from expected one ('{}')".format(str(t), string))


@pytest.mark.parametrize('init_value', ['gielie', '03 Mar 2003'])
def test_construct_invalid_timestamp(init_value):
    with pytest.raises(ValueError):
        katpoint.Timestamp(init_value)


def test_numerical_timestamp():
    """Test numerical properties of timestamps."""
    t0 = 1248186982.3980861
    t = katpoint.Timestamp(t0)
    assert t == t + 0.0
    assert t != t + 1.0
    assert t > t - 1.0
    assert t < t + 1.0
    assert t == eval('katpoint.' + repr(t))
    assert float(t) == t0
    t1 = Time('2009-07-21 02:52:12.34')
    t = katpoint.Timestamp(t1)
    assert t.time == t1
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
