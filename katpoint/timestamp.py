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
from __future__ import print_function, division, absolute_import
from builtins import object
from past.builtins import basestring

import time
import math

from functools import total_ordering

import numpy as np
from astropy.time import Time


@total_ordering
class Timestamp(object):
    """Basic representation of time, in UTC seconds since Unix epoch.

    This is loosely based on :class:`ephem.Date`. Its base representation
    of time is UTC seconds since the Unix epoch, i.e. the standard Posix
    timestamp. Fractional seconds are allowed, as the basic data
    type is a Python (double-precision) float.

    The following input formats are accepted for a timestamp:

    - None, which uses the current time (the default).

    - A floating-point number, directly representing the number of UTC seconds
      since the Unix epoch. Fractional seconds are allowed.

    - A string with format 'YYYY-MM-DD HH:MM:SS.SSS' or 'YYYY/MM/DD HH:MM:SS.SSS',
      or any prefix thereof. Examples are '1999-12-31 12:34:56.789', '1999-12-31',
      '1999-12-31 12:34:56' and even '1999'. The input string is always in UTC.

    - A :class:`astropy.Time.time` object.

    Parameters
    ----------
    timestamp : float, string, :class:`astropy.Time.time` object or None
        Timestamp, in various formats (if None, defaults to now)

    Arguments
    ---------
    secs : float
        Timestamp as UTC seconds since Unix epoch
    """

    def __init__(self, timestamp=None):
        if isinstance(timestamp, basestring):
            try:
                timestamp = timestamp.strip().replace('/', '-')
                timestamp = Time(decode(timestamp))
            except ValueError:
                raise ValueError("Timestamp string '%s' not in correct format - " % (timestamp,) +
                                 "should be 'YYYY-MM-DD HH:MM:SS' or 'YYYY/MM/DD HH:MM:SS' or prefix thereof " +
                                 "(all UTC, fractional seconds allowed)")
        if timestamp is None:
            self.secs = time.time()
        elif isinstance(timestamp, Time):
            iso = timestamp.iso
            timestamp = [int(iso[:4]), int(iso[5:7]), int(iso[8:10]), int(iso[11:13]), int(iso[14:16]), float(iso[17:])]
            timestamp = timestamp + [0, 0, 0]
            int_secs = math.floor(timestamp[5])
            frac_secs = timestamp[5] - int_secs
            timestamp[5] = int(int_secs)
            self.secs = time.mktime(tuple(timestamp)) - time.timezone + frac_secs
        else:
            self.secs = float(timestamp)

    # Keep object small by using __slots__ instead of __dict__
    __slots__ = 'secs'

    def __repr__(self):
        """Short machine-friendly string representation of timestamp object."""
        return 'Timestamp(%s)' % repr(self.secs)

    def __str__(self):
        """Verbose human-friendly string representation of timestamp object."""
        return self.to_string()

    def __eq__(self, other):
        """Test for equality"""
        return self.secs == float(other)

    def __lt__(self, other):
        """Test for less than"""
        return self.secs < float(other)

    def __add__(self, other):
        """Add seconds (as floating-point number) to timestamp and return result."""
        return Timestamp(self.secs + other)

    def __sub__(self, other):
        """
            Subtract seconds (floating-point number is treated as a time interval) from timestamp
            and return result. If used for the difference between two (absolute time) Timestamps
            then the result is an interval in seconds (a floating-point number).
        """
        if isinstance(other, Timestamp):
            return self.secs - other.secs
        else:
            return Timestamp(self.secs - other)

    def __mul__(self, other):
        """Multiply timestamp by numerical factor (useful for processing timestamps)."""
        return Timestamp(self.secs * other)

    def __div__(self, other):
        """Divide timestamp by numerical factor (useful for processing timestamps)."""
        return Timestamp(self.secs / other)

    def __truediv__(self, other):
        """Divide timestamp by numerical factor (useful for processing timestamps)."""
        return Timestamp(self.secs / other)

    def __radd__(self, other):
        """Add timestamp to seconds (as floating-point number) and return result."""
        return Timestamp(other + self.secs)

    def __iadd__(self, other):
        """Add seconds (as floating-point number) to timestamp in-place."""
        self.secs += other
        return self

    def __rsub__(self, other):
        """
            Subtract timestamp from seconds (as floating-point number) and return
            resulting seconds (floating-point number). This is typically used when
            calculating the interval between two absolute instants of time.
        """
        return other - self.secs

    def __isub__(self, other):
        """Subtract seconds (as floating-point number) from timestamp in-place."""
        self.secs -= other
        return self

    def __float__(self):
        """Convert to floating-point UTC seconds."""
        return self.secs

    def __hash__(self):
        """Base hash on internal timestamp, just like equality operator."""
        return hash(self.secs)

    def local(self):
        """Convert timestamp to local time string representation (for display only)."""
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
        int_secs = math.floor(self.secs)
        frac_secs = np.round(1000.0 * (self.secs - int_secs)) / 1000.0
        if frac_secs >= 1.0:
            int_secs += 1.0
            frac_secs -= 1.0
        datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(int_secs))
        if frac_secs == 0.0:
            return datetime
        else:
            return '%s%5.3f' % (datetime[:-1], float(datetime[-1]) + frac_secs)

    def to_ephem_date(self):
        """Convert timestamp to :class:`astropy.time.Time` object."""
        int_secs = math.floor(self.secs)
        timetuple = list(time.gmtime(int_secs)[:6])
        timetuple[5] += self.secs - int_secs
        return Time('{0}-{1:02}-{2:02} {3:02}:{4:02}:{5:02}'.format(*timetuple))

    def to_mjd(self):
        """Convert timestamp to Modified Julian Day (MJD)."""
        djd = self.to_ephem_date()
        return djd.mjd


def decode(s):
    """Decodes a date string
    """
    # Look for a dot
    dot = s.find('.')
    if dot > 0:
        # fractional part of the seconds
        f = s[dot:]
        # date/time without fractional seconds
        s = s[:dot]
    else:
        f = '.0'

    # time without fractional seconds
    try:
        d = time.strptime(s, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        try:
            d = time.strptime(s, '%Y-%m-%d %H:%M')
        except ValueError:
            try:
                d = time.strptime(s, '%Y-%m-%d %H')
            except ValueError:
                try:
                    d = time.strptime(s, '%Y-%m-%d')
                except ValueError:
                    try:
                        d = time.strptime(s, '%Y-%m')
                    except ValueError:
                        try:
                            d = time.strptime(s, '%Y')
                        except ValueError:
                            raise ValueError('unable to decode date string')

    # Convert to a unix time and add the fractional seconds
    u = time.mktime(d)

    # Back to a tuple
    d = time.localtime(u)

    return time.strftime('%Y-%m-%d %H:%M:%S', d) + f


def now():
    """ Create a Date representing 'now'
    """
    return Date(Time.now().mjd - _djd)
