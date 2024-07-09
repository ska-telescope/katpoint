################################################################################
# Copyright (c) 2009-2010,2013,2017-2023, National Research Foundation (SARAO)
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

import re
import warnings
from unittest.mock import patch

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time, TimeDelta

try:
    from astropy.time import TimeDeltaMissingUnitWarning
except ImportError:
    TimeDeltaMissingUnitWarning = None

import katpoint


@pytest.mark.parametrize(
    "init_value, string",
    [
        (1248186982.3980861, "2009-07-21 14:36:22.398"),
        (Time("2009-07-21 02:52:12.34"), "2009-07-21 02:52:12.340"),
        (0, "1970-01-01 00:00:00.000"),
        (-10, "1969-12-31 23:59:50.000"),
        ("2009-07-21 02:52:12.034", "2009-07-21 02:52:12.034"),
        ("2009-07-21 02:52:12.000", "2009-07-21 02:52:12.000"),
        ("2009-07-21 02:52:12", "2009-07-21 02:52:12.000"),
        ("2009-07-21 02:52", "2009-07-21 02:52:00.000"),
        ("2009-07-21", "2009-07-21 00:00:00.000"),
        ("2009/07/21 02:52:12.034", "2009-07-21 02:52:12.034"),
        ("2009/07/21 02:52:12.000", "2009-07-21 02:52:12.000"),
        ("2009/07/21 02:52:12", "2009-07-21 02:52:12.000"),
        ("2009/07/21 02:52", "2009-07-21 02:52:00.000"),
        (b"2009/07/21", "2009-07-21 00:00:00.000"),
        (b"2020-07-17 12:40:12", "2020-07-17 12:40:12.000"),
    ],
)
def test_construct_valid_timestamp(init_value, string):
    """Test that we can construct a `Timestamp` from valid inputs."""
    t = katpoint.Timestamp(init_value)
    assert (
        str(t) == string
    ), f"Timestamp string ('{str(t)}') differs from expected one ('{string}')"
    # Exercise local() code path too
    print(t.local())


@pytest.mark.parametrize(
    "init_value",
    [
        "gielie",
        "03 Mar 2003",
        2j,  # An unsupported NumPy dtype
        "2020-07-28T18:18:18.000",  # ISO 8601 with a 'T' is invalid
        "2020-07-28 18:18:18.000+02:00",  # Time zones are not accepted
        TimeDelta(
            1.0, format="sec", scale="tai"
        ),  # A TimeDelta is the wrong kind of Time
    ],
)
def test_construct_invalid_timestamp(init_value):
    """Test that `Timestamp` rejects invalid inputs."""
    with pytest.raises(ValueError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.ComplexWarning)
            katpoint.Timestamp(init_value)


def test_string_representations():
    """Test string representations of `Timestamp`."""
    t = katpoint.Timestamp(1234567890.01234)
    assert t.to_string() == "2009-02-13 23:31:30.012"
    assert str(t) == "2009-02-13 23:31:30.012"
    assert repr(t) == "Timestamp(1234567890.01234)"
    # XXX We could mock time.localtime to control the output of Timestamp.local()
    assert len(t.local().split()) == 3
    assert t.local().split()[1].endswith("30.012")
    # Change the output precision
    t.time.precision = 5
    assert str(t) == "2009-02-13 23:31:30.01234"
    assert t.local().split()[1].endswith("30.01234")


def test_current_timestamp():
    """Check that `Timestamp()` is equal to `Time.now()`."""
    t0 = Time.now()
    with patch.object(Time, "now", return_value=t0):
        assert katpoint.Timestamp() == t0


def test_timestamp_ordering():
    """Check that `Timestamp`s can be compared to each other."""
    t1 = katpoint.Timestamp(1234567890)
    t2 = katpoint.Timestamp(1234567891)
    assert t2 >= t1
    assert t2 >= t1.time
    assert t2 >= t1.secs
    assert t2 > t1
    assert t1 <= t2
    assert t1 < t2


def test_numerical_timestamp():
    """Test numerical properties of timestamps."""
    t0 = 1248186982.3980861
    t = katpoint.Timestamp(t0)
    assert t == t + 0.0
    assert t != t + 1.0
    assert t > t - 1.0
    assert t < t + 1.0
    # This only works for scalars...
    repr_float = float(re.match(r"^Timestamp\((.*)\)$", repr(t)).group(1))
    assert t == katpoint.Timestamp(repr_float)
    assert float(t) == t0

    # XXX Reimplement Astropy 4.2's Time.isclose for now
    # to avoid depending on Python 3.7.
    atol = 2 * np.finfo(float).eps * u.day
    t1 = Time("2009-07-21 02:52:12.34")
    t = katpoint.Timestamp(t1)
    t += 2.0
    t -= 2.0
    assert abs(t.time - t1) <= atol
    t += 2.0 * u.year
    t -= 2.0 * u.year
    assert abs(t.time - t1) <= atol
    t2 = t + 1 * u.day
    assert (t2 - t) << u.second == 1 * u.day
    assert t / 2.0 == t * 0.5
    assert 1.0 + t == t + 1.0
    assert abs((t - 1.0 * u.day).time - (t1 - 1.0 * u.day)) < atol
    try:
        assert hash(t) == hash(t + 0.0), "Timestamp hashes not equal"
    except TypeError:
        pytest.fail("Timestamp object not hashable")


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

    # Check various additions and subtractions
    def approx_equal(x, y, **kwargs):
        return x.secs == pytest.approx(y, **kwargs)

    # Timestamp + interval
    assert approx_equal(t + 1, t0 + 1)
    assert approx_equal(t + 1 * u.second, t0 + 1)
    assert approx_equal(t + TimeDelta(1.0, format="sec", scale="tai"), t0 + 1)
    # interval + Timestamp
    assert approx_equal(1 + t, t0 + 1)
    with pytest.raises(TypeError):  # why does Quantity not return NotImplemented here?
        assert approx_equal(1 * u.second + t, t0 + 1)
    with warnings.catch_warnings():
        # Ignore any TimeDeltaMissingUnitWarning here, since this is triggered
        # in passing by TimeDelta.__add__(time_delta, t) on the way to
        # NotImplemented, which eventually results in the desired call
        # Timestamp.__radd__(t, time_delta).
        # XXX Reconsider this workaround once we depend on Astropy >= 6.0
        warnings.simplefilter("ignore", TimeDeltaMissingUnitWarning)
        assert approx_equal(TimeDelta(1.0, format="sec", scale="tai") + t, t0 + 1)
    # Timestamp + Timestamp
    with pytest.raises(ValueError):
        print(t + t)
    with pytest.raises(ValueError):
        print(t + t.time)
    # Timestamp - interval
    assert approx_equal(t - 1, t0 - 1)
    assert approx_equal(t - 1 * u.second, t0 - 1)
    assert approx_equal(t - TimeDelta(1.0, format="sec", scale="tai"), t0 - 1)
    # This differs from PyEphem-based katpoint: leap seconds!
    assert approx_equal(t - t0, 26.0, rel=1e-5)  # float t0 is an interval here...
    # Timestamp - Timestamp
    assert t - katpoint.Timestamp(t0) == 0.0
    assert t - t.time == 0.0
    assert t0 - t == 0.0  # float t0 is a Unix timestamp here...
    with warnings.catch_warnings():
        # Ignore any TimeDeltaMissingUnitWarning here, since this is triggered
        # in passing by Time.__sub__(t.time, t) on the way to NotImplemented, which
        # eventually results in the desired call Timestamp.__rsub__(t, t.time).
        # XXX Reconsider this workaround once we depend on Astropy >= 6.0
        warnings.simplefilter("ignore", TimeDeltaMissingUnitWarning)
        assert t.time - t == 0.0
    # Timestamp += interval
    t += 1
    assert approx_equal(t, t0 + 1)
    t += 1 * u.second
    assert approx_equal(t, t0 + 2)
    t += TimeDelta(1.0, format="sec", scale="tai")
    assert approx_equal(t, t0 + 3)
    # Timestamp += Timestamp
    with pytest.raises(ValueError):
        t += t
    with pytest.raises(ValueError):
        t += t.time
    # Timestamp -= interval
    t -= 1
    assert approx_equal(t, t0 + 2)
    t -= 1 * u.second
    assert approx_equal(t, t0 + 1)
    t -= TimeDelta(1.0, format="sec", scale="tai")
    assert approx_equal(t, t0)
    # Timestamp -= Timestamp
    with pytest.raises(ValueError):
        t -= t
    with pytest.raises(ValueError):
        t -= t.time


def test_array_timestamps():
    """Test that `Timestamp` supports arrays of times."""
    t = katpoint.Timestamp([1234567890.0, 1234567891.0])
    with pytest.raises(TypeError):
        float(t)
    np.testing.assert_array_equal(t == 1234567890.0, [True, False])
    np.testing.assert_array_equal(t != 1234567890.0, [False, True])
    t2 = katpoint.Timestamp(1234567890.0)
    t_array_1d = t2 + np.arange(3)
    np.testing.assert_array_equal(
        t_array_1d.to_string(),
        [
            "2009-02-13 23:31:30.000",
            "2009-02-13 23:31:31.000",
            "2009-02-13 23:31:32.000",
        ],
    )
    assert repr(t_array_1d) == "Timestamp([1234567890.000 ... 1234567892.000])"
    assert t_array_1d.local().shape == (3,)
    # Construct from sequence or array of strings or `Time`s or `Timestamp`s
    t0 = katpoint.Timestamp(t.time[0])
    t1 = katpoint.Timestamp(t.time[1])
    t3 = katpoint.Timestamp([str(t0), str(t1)])
    assert t3.time.shape == (2,)
    assert all(t3 == t)
    t4 = katpoint.Timestamp((t0.time, t1.time))
    assert t4.time.shape == (2,)
    assert all(t4 == t)
    t5 = katpoint.Timestamp(np.array((t0, t1)))
    assert t5.time.shape == (2,)
    assert all(t5 == t)
    # Construct from 2-dimensional array of floats or a 2-D `Time`
    array_2d = [[1234567890.0, 1234567891.0], [1234567892.0, 1234567893.0]]
    t_array_2d = katpoint.Timestamp(array_2d)
    t_array_2d = katpoint.Timestamp(t_array_2d.time)
    np.testing.assert_array_equal(t_array_2d.secs, array_2d)


@pytest.mark.parametrize("mjd", [59000.0, (59000.0, 59001.0)])
def test_mjd_timestamp(mjd):
    """Test `Timestamp.to_mjd` functionality."""
    t = Time(mjd, format="mjd")
    t2 = katpoint.Timestamp(t)
    assert np.all(t2.to_mjd() == mjd)
