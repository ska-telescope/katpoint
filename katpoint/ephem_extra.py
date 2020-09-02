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

"""Enhancements to PyEphem."""

import numpy as np
import astropy.units as u
from astropy.coordinates import Angle

# --------------------------------------------------------------------------------------------------
# --- Helper functions
# --------------------------------------------------------------------------------------------------

# The speed of light, in metres per second
lightspeed = 299792458.0


def is_iterable(x):
    """Checks if object is iterable (but not a string or 0-dimensional array)."""
    return hasattr(x, '__iter__') and not isinstance(x, str) and \
        not (getattr(x, 'shape', None) == ())


def rad2deg(x):
    """Converts radians to degrees (also works for arrays)."""
    return x * (180.0 / np.pi)


def deg2rad(x):
    """Converts degrees to radians (also works for arrays)."""
    return x * (np.pi / 180.0)


def _just_gimme_an_ascii_string(s):
    """Converts encoded/decoded string to a platform-appropriate ASCII string.

    On Python 2 this encodes Unicode strings to normal ASCII strings, while
    normal strings are left unchanged. On Python 3 this decodes bytes to
    Unicode strings via the ASCII encoding, while Unicode strings are left
    unchanged (and might still contain non-ASCII characters!).

    Raises
    ------
    UnicodeEncodeError, UnicodeDecodeError
        If the conversion fails due to the presence of non-ASCII characters
    """
    if isinstance(s, bytes) and not isinstance(s, str):
        # Only encoded bytes on Python 3 will end up here
        return str(s, encoding='ascii')
    else:
        return str(s)


def angle_from_degrees(s):
    """Creates angle object from sexagesimal string in degrees or number in radians."""
    try:
        return Angle(s)
    except u.UnitsError:
        # Deal with user input
        if isinstance(s, bytes):
            s = s.decode(encoding='ascii')
        # We now have a number, string or tuple without a unit
        if isinstance(s, (str, tuple)):
            return Angle(s, unit=u.deg)
        else:
            return Angle(s, unit=u.rad)


def angle_from_hours(s):
    """Creates angle object from sexagesimal string in hours or number in radians."""
    try:
        return Angle(s)
    except u.UnitsError:
        # Deal with user input
        if isinstance(s, bytes):
            s = s.decode(encoding='ascii')
        # We now have a number, string or tuple without a unit
        if isinstance(s, str) and ':' in s or isinstance(s, tuple):
            return Angle(s, unit=u.hour)
        if isinstance(s, str):
            return Angle(s, unit=u.deg)
        else:
            return Angle(s, unit=u.rad)


def wrap_angle(angle, period=2.0 * np.pi):
    """Wrap angle into interval centred on zero.

    This wraps the *angle* into the interval -*period* / 2 ... *period* / 2.
    """
    return (angle + 0.5 * period) % period - 0.5 * period
