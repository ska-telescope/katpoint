################################################################################
# Copyright (c) 2009-2013,2017-2021,2023, National Research Foundation (SARAO)
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

"""A Timestamp object."""

import time

import numpy as np
from astropy.time import Time, TimeDelta


def delta_seconds(x):
    """Construct a `TimeDelta` in TAI seconds."""
    return TimeDelta(x, format="sec", scale="tai")


class Timestamp:
    """Basic representation of time(s), in UTC seconds since Unix epoch.

    This is loosely based on PyEphem's `Date` object, but uses an Astropy
    `Time` object as internal representation. Like `Time` it can contain
    a multi-dimensional array of timestamps.

    The following input formats are accepted for a timestamp:

    - A floating-point number, directly representing the number of UTC seconds
      since the Unix epoch. Fractional seconds are allowed.

    - A string or bytes with format 'YYYY-MM-DD HH:MM:SS.SSS' (Astropy 'iso'
      format) or 'YYYY/MM/DD HH:MM:SS.SSS' (XEphem format), where the hours
      and minutes, seconds, and fractional seconds are optional. It is always
      in UTC. Examples are:

        '1999-12-31 12:34:56.789'
        '1999/12/31 12:34:56'
        '1999-12-31 12:34'
        b'1999-12-31'

    - A :class:`~astropy.time.Time` object (NOT :class:`~astropy.time.TimeDelta`).

    - Another :class:`Timestamp` object, which will result in a copy.

    - A sequence or NumPy array of one of the above types.

    - None, which uses the current time (the default).

    Parameters
    ----------
    timestamp : :class:`~astropy.time.Time`, :class:`Timestamp`, float, string,
                bytes, sequence or array of any of the former, or None, optional
        Timestamp, in various formats (if None, defaults to now)

    Raises
    ------
    ValueError
        If `timestamp` is not in a supported format

    Attributes
    ----------
    time : :class:`~astropy.time.Time`
        Underlying `Time` object
    secs : float or array of float
        Timestamp as UTC seconds since Unix epoch

    Notes
    -----
    This differs from :class:`~astropy.time.Time` in the following respects:

    - Numbers are interpreted as Unix timestamps during initialisation;
      `Timestamp(1234567890)` is equivalent to `Time(1234567890, format='unix')`
      (while `Time(1234567890)` is not allowed because it lacks a format).

    - Arithmetic is done in seconds instead of days (in the absence of units).

    - Date strings may contain slashes (a leftover from PyEphem / XEphem).

    - Empty initialisation results in the current time, so `Timestamp()`
      is equivalent to `Time.now()` (while `Time()` is not allowed).
    """

    def __init__(self, timestamp=None):
        if timestamp is None:
            self.time = Time.now()
        elif isinstance(timestamp, Timestamp):
            self.time = timestamp.time.replicate()
        elif isinstance(timestamp, TimeDelta):
            raise ValueError(f"Cannot construct Timestamp from TimeDelta {timestamp}")
        elif isinstance(timestamp, Time):
            self.time = timestamp.replicate()
        else:
            # Convert to array to simplify both array/scalar and string/bytes handling
            val = np.asarray(timestamp)
            # Turn array of Timestamps into array of corresponding internal Time objects
            if val.size > 0 and isinstance(val.flat[0], Timestamp):
                val = np.vectorize(lambda ts: ts.time)(val)
            time_format = None
            if val.dtype.kind == "U":
                # Convert default PyEphem timestamp strings to ISO strings
                val = np.char.replace(np.char.strip(val), "/", "-")
                time_format = "iso"
            elif val.dtype.kind == "S":
                val = np.char.replace(np.char.strip(val), b"/", b"-")
                time_format = "iso"
            elif val.dtype.kind in "iuf":
                # Consider any number to be a Unix timestamp
                time_format = "unix"
            self.time = Time(val, format=time_format, scale="utc", precision=3)

    @property
    def secs(self):
        """Timestamp as UTC seconds since Unix epoch."""
        return self.time.utc.unix

    def __repr__(self):
        """Short machine-friendly string representation of timestamp object."""
        # We need a custom formatter because suppress=True only works on values < 1e8
        # and today's Unix timestamps are bigger than that
        formatter = f"{{:.{self.time.precision:d}f}}".format
        with np.printoptions(threshold=2, edgeitems=1, formatter={"float": formatter}):
            return f"Timestamp({self.secs})"

    def __str__(self):
        """Verbose human-friendly string representation of timestamp object."""
        return str(self.to_string())

    def __eq__(self, other):
        """Test for equality."""
        return self.time == Timestamp(other).time

    def __ne__(self, other):
        """Test for inequality."""
        return self.time != Timestamp(other).time

    def __lt__(self, other):
        """Test for less than."""
        return self.time < Timestamp(other).time

    def __le__(self, other):
        """Test for less than or equal to."""
        return self.time <= Timestamp(other).time

    def __gt__(self, other):
        """Test for greater than."""
        return self.time > Timestamp(other).time

    def __ge__(self, other):
        """Test for greater than or equal to."""
        return self.time >= Timestamp(other).time

    def __add__(self, other):
        """Add seconds (as floating-point number) to timestamp and return result."""
        return Timestamp(self.time + delta_seconds(other))

    def __sub__(self, other):
        """Subtract seconds (floating-point time interval) from timestamp.

        If used for the difference between two (absolute time) Timestamps
        then the result is an interval in seconds (a floating-point number).
        """
        if isinstance(other, Timestamp):
            return (self.time - other.time).sec
        elif isinstance(other, Time) and not isinstance(other, TimeDelta):
            return (self.time - other).sec
        else:
            return Timestamp(self.time - delta_seconds(other))

    def __mul__(self, other):
        """Multiply timestamp by numerical factor (useful for processing timestamps)."""
        return Timestamp(self.secs * other)

    def __truediv__(self, other):
        """Divide timestamp by numerical factor (useful for processing timestamps)."""
        return Timestamp(self.secs / other)

    def __radd__(self, other):
        """Add timestamp to seconds (as floating-point number) and return result."""
        return Timestamp(self.time + delta_seconds(other))

    def __iadd__(self, other):
        """Add seconds (as floating-point number) to timestamp in-place."""
        self.time += delta_seconds(other)
        return self

    def __rsub__(self, other):
        """Subtract timestamp from seconds (as floating-point number).

        Return resulting seconds (floating-point number). This is typically
        used when calculating the interval between two absolute instants
        of time.
        """
        return (Timestamp(other).time - self.time).sec

    def __isub__(self, other):
        """Subtract seconds (as floating-point number) from timestamp in-place."""
        self.time -= delta_seconds(other)
        return self

    def __float__(self):
        """Convert scalar timestamp to floating-point UTC seconds."""
        try:
            return float(self.secs)
        except TypeError as err:
            raise TypeError(
                "Float conversion only supported for scalar Timestamps"
            ) from err

    def __hash__(self):
        """Compute hash on internal timestamp, just like equality operator."""
        return hash(self.time)

    def local(self):
        """Local time string representation (str or array of str)."""
        prec = self.time.precision
        frac_secs, int_secs = np.modf(np.round(self.secs, decimals=prec))

        def local_time_string(f, i):
            format_string = f"%Y-%m-%d %H:%M:%S.{f * 10**prec:0{prec}.0f} %Z"
            return time.strftime(format_string, time.localtime(i))

        local_str = np.vectorize(local_time_string)(frac_secs, int_secs)
        return local_str if local_str.ndim else local_str.item()

    def to_string(self):
        """UTC string representation (str or array of str)."""
        return self.time.iso

    def to_mjd(self):
        """Convert timestamp to Modified Julian Day (MJD)."""
        return self.time.mjd
