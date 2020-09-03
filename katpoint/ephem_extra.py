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


def wrap_angle(angle, period=2.0 * np.pi):
    """Wrap angle into interval centred on zero.

    This wraps the *angle* into the interval -*period* / 2 ... *period* / 2.
    """
    return (angle + 0.5 * period) % period - 0.5 * period
