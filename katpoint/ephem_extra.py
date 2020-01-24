################################################################################
# Copyright (c) 2009-2019, National Research Foundation (Square Kilometre Array)
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
from __future__ import print_function, division, absolute_import
from builtins import object
from past.builtins import basestring

import numpy as np
from astropy.coordinates import Angle
from astropy.coordinates import AltAz
from astropy.coordinates import ICRS
from astropy import units

# --------------------------------------------------------------------------------------------------
# --- Helper functions
# --------------------------------------------------------------------------------------------------

# The speed of light, in metres per second
lightspeed = 299792458.0


def is_iterable(x):
    """Checks if object is iterable (but not a string or 0-dimensional array)."""
    return hasattr(x, '__iter__') and not isinstance(x, basestring) and \
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
        # Ephem expects a number or platform-appropriate string (i.e. Unicode on Py3)
        if type(s) == str:
            return Angle(s, unit=units.deg)
        elif type(s) == tuple:
            return Angle(s, unit=units.deg)
        else:
            return Angle(s, unit=units.rad)
    except TypeError:
        # If input is neither, assume that it really wants to be a string
        return Angle(_just_gimme_an_ascii_string(s), unit=units.deg)


def angle_from_hours(s):
    """Creates angle object from sexagesimal string in hours or number in radians."""
    try:
        # Ephem expects a number or platform-appropriate string (i.e. Unicode on Py3)
        if type(s) == str:
            return Angle(s, unit=units.hour)
        elif type(s) == tuple:
            return Angle(s, unit=units.hour)
        else:
            return Angle(s, unit=units.rad)
    except TypeError:
        # If input is neither, assume that it really wants to be a string
        return Angle(_just_gimme_an_ascii_string(s), unit=units.hour)


def wrap_angle(angle, period=2.0 * np.pi):
    """Wrap angle into interval centred on zero.

    This wraps the *angle* into the interval -*period* / 2 ... *period* / 2.

    """
    return (angle + 0.5 * period) % period - 0.5 * period

# --------------------------------------------------------------------------------------------------
# --- CLASS :  StationaryBody
# --------------------------------------------------------------------------------------------------


class StationaryBody(object):
    """Stationary body with fixed (az, el) coordinates.

    This is a simplified :class:`Body` that is useful to specify targets
    such as zenith and geostationary satellites.

    Parameters
    ----------
    az, el : string or float
        Azimuth and elevation, either in 'D:M:S' string format, or float in rads
    name : string, optional
        Name of body

    """
    def __init__(self, az, el, name=None):
        self.az = angle_from_degrees(az)
        self.el = angle_from_degrees(el)
        self.alt = self.el  # alternative terminology
        if not name:
            name = "Az: %s El: %s" % (self.az, self.el)
        self.name = name

    def compute(self, loc, date, pressure):
        """Update target coordinates for given observer.

        This updates the (ra, dec) coordinates of the target, as seen from the
        given *observer*, while its (az, el) coordinates remain unchanged.

        """
        altaz = AltAz(alt=self.el, az=self.az, location=loc,
                obstime=date, pressure=pressure)
        radec = altaz.transform_to(ICRS)
        self.ra = radec.ra
        self.dec = radec.dec
        # This is a kludge, as XEphem provides no way to convert apparent
        # (ra, dec) back to astrometric (ra, dec)
        self.a_ra = radec.ra
        self.a_dec = radec.dec

# --------------------------------------------------------------------------------------------------
# --- CLASS :  NullBody
# --------------------------------------------------------------------------------------------------


class NullBody(object):
    """Body with no position, used as a placeholder.

    This body has the expected methods of :class:`Body`, but always returns NaNs
    for all coordinates. It is intended for use as a placeholder when no proper
    target object is available, i.e. as a dummy target.

    """
    def __init__(self):
        self.name = 'Nothing'
        self.az = self.alt = self.el = np.nan
        self.ra = self.dec = self.a_ra = self.a_dec = np.nan

    def compute(self, loc, date, pressure):
        pass
