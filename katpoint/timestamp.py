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

"""A Timestamp object."""

import time
import math

import numpy as np
import astropy.time
import astropy.units as u
from astropy.time import Time


class Timestamp:
    """Basic representation of time, in UTC seconds since Unix epoch.

    This is loosely based on PyEphem's `Date` object. Its base representation
    of time is UTC seconds since the Unix epoch, i.e. the standard Posix
    timestamp. Fractional seconds are allowed, as the basic data
    type is a Python (double-precision) float.

    The following input formats are accepted for a timestamp:

    - None, which uses the current time (the default).

    - A floating-point number, directly representing the number of UTC seconds
      since the Unix epoch. Fractional seconds are allowed.

    - A string or bytes with format 'YYYY-MM-DD HH:MM:SS.SSS' or
      'YYYY/MM/DD HH:MM:SS.SSS', where the hours and minutes, seconds, and
      fractional seconds are optional. It is always in UTC. Examples are:

        '1999-12-31 12:34:56.789'
        '1999/12/31 12:34:56'
        '1999-12-31 12:34'
        b'1999-12-31'

    - A :class:`~astropy.time.Time` object.

    - A :class:`Timestamp` object, which will result in a shallow copy.

    Parameters
    ----------
    timestamp : float, string, bytes, :class:`~astropy.time.Time`, :class:`Timestamp` or None
        Timestamp, in various formats (if None, defaults to now)

    Arguments
    ---------
    secs : float
        Timestamp as UTC seconds since Unix epoch
    """

    def __init__(self, timestamp=None):
        format = None
        if timestamp is None:
            self.time = Time.now()
        elif isinstance(timestamp, Timestamp):
            self.time = timestamp.time
        else:
            # Use Astropy internal function to cast input to float64,
            # string (unicode / bytes) or Time object array (0-dim for scalar)
            val = astropy.time.core._make_array(timestamp)
            format = None
            if val.dtype.kind == 'U':
                # Convert default PyEphem timestamp strings to ISO strings
                val = np.char.replace(np.char.strip(val), '/', '-')
                format = 'iso'
            elif val.dtype.kind == 'S':
                val = np.char.replace(np.char.strip(val), b'/', b'-')
                format = 'iso'
            elif val.dtype.kind == 'f':
                format = 'unix'
            self.time = Time(val, format=format, scale='utc', precision=3)

    @property
    def secs(self):
        return self.time.utc.unix

    def __repr__(self):
        """Short machine-friendly string representation of timestamp object."""
        t = self.secs
        if t.shape in {(), (0,)}:
            return 'Timestamp({})'.format(t)
        elif t.shape == (1,):
            return 'Timestamp([{!r}])'.format(t[0])
        elif t.shape == (2,):
            return 'Timestamp([{!r}, {!r}])'.format(t[0], t[-1])
        else:
            return 'Timestamp([{!r}, ..., {!r}])'.format(t[0], t[-1])

    def __str__(self):
        """Verbose human-friendly string representation of timestamp object."""
        return self.to_string()

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
        return Timestamp(self.time + (other << u.second))

    def __sub__(self, other):
        """Subtract seconds (floating-point time interval) from timestamp.

        If used for the difference between two (absolute time) Timestamps
        then the result is an interval in seconds (a floating-point number).
        """
        if isinstance(other, Timestamp):
            return (self.time - other.time).sec
        elif isinstance(other, Time):
            return (self.time - other).sec
        else:
            return Timestamp(self.time - (other << u.second))

    def __mul__(self, other):
        """Multiply timestamp by numerical factor (useful for processing timestamps)."""
        return Timestamp(self.secs * other)

    def __truediv__(self, other):
        """Divide timestamp by numerical factor (useful for processing timestamps)."""
        return Timestamp(self.secs / other)

    def __radd__(self, other):
        """Add timestamp to seconds (as floating-point number) and return result."""
        return Timestamp(self.time + (other << u.second))

    def __iadd__(self, other):
        """Add seconds (as floating-point number) to timestamp in-place."""
        self.time += (other << u.second)
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
        self.time -= (other << u.second)
        return self

    def __float__(self):
        """Convert scalar timestamp to floating-point UTC seconds."""
        try:
            return float(self.secs)
        except TypeError as err:
            raise TypeError('Float conversion only supported for scalar Timestamps') from err

    def __hash__(self):
        """Base hash on internal timestamp, just like equality operator."""
        return hash(self.time)

    def local(self):
        """Convert scalar timestamp to local time string representation (for display only)."""
        if self.time.shape != ():
            raise TypeError('String output only supported for scalar Timestamps')
        int_secs = math.floor(self.secs)
        frac_secs = np.round(1000.0 * (self.secs - int_secs)) / 1000.0
        if frac_secs >= 1.0:
            int_secs += 1.0
            frac_secs -= 1.0
        datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int_secs))
        timezone = time.strftime('%Z', time.localtime(int_secs))
        if frac_secs == 0.0:
            return '%s %s' % (datetime, timezone)
        else:
            return '%s%5.3f %s' % (datetime[:-1], float(datetime[-1]) + frac_secs, timezone)

    def to_string(self):
        """Convert timestamp to UTC string representation."""
        return str(self.time.iso)

    def to_mjd(self):
        """Convert timestamp to Modified Julian Day (MJD)."""
        return self.time.mjd
