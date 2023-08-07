################################################################################
# Copyright (c) 2009-2022, National Research Foundation (Square Kilometre Array)
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

"""
Coordinate library for the MeerKAT project, providing astronomical coordinate
transformations, antenna pointing models, correlator delay models, source flux
models and basic source catalogues, using an existing coordinate library such
as Astropy to do the low-level calculations.
"""

import logging as _logging
from types import ModuleType as _ModuleType

import numpy as _np

from .antenna import Antenna
from .catalogue import Catalogue, specials
from .conversion import (
    azel_to_enu,
    ecef_to_enu,
    ecef_to_lla,
    enu_to_azel,
    enu_to_ecef,
    enu_to_xyz,
    hadec_to_enu,
    lla_to_ecef,
)
from .delay_correction import DelayCorrection
from .delay_model import DelayModel
from .flux import FluxDensityModel, FluxError
from .model import BadModelFile, Model, Parameter
from .pointing import PointingModel
from .projection import plane_to_sphere, sphere_to_plane
from .refraction import RefractionCorrection
from .target import (
    NonAsciiError,
    Target,
    construct_azel_target,
    construct_radec_target,
)
from .timestamp import Timestamp


def wrap_angle(angle, period=2.0 * _np.pi):
    """Wrap angle into interval centred on zero.

    This wraps the *angle* into the interval -*period* / 2 ... *period* / 2.
    """
    return (angle + 0.5 * period) % period - 0.5 * period


# Setup library logger and add a print-like handler used when no logging is configured
class _NoConfigFilter(_logging.Filter):
    """Filter which only allows event if top-level logging is not configured."""

    def filter(self, record):
        return 1 if not _logging.root.handlers else 0


_no_config_handler = _logging.StreamHandler()
_no_config_handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT))
_no_config_handler.addFilter(_NoConfigFilter())
logger = _logging.getLogger(__name__)
logger.addHandler(_no_config_handler)

# Document the public API in __all__ / __dir__ by discarding modules and private variables
__all__ = [
    n
    for n, o in globals().items()
    if not isinstance(o, _ModuleType) and not n.startswith("_")
]
__all__ += ["__version__"]


def __dir__():
    """IPython tab completion seems to respect this."""
    return __all__


# BEGIN VERSION CHECK
# Get package version when locally imported from repo or via -e develop install
try:
    import katversion as _katversion
except ImportError:
    import time as _time

    __version__ = "0.0+unknown.{}".format(_time.strftime("%Y%m%d%H%M"))
else:
    __version__ = _katversion.get_version(__path__[0])
# END VERSION CHECK
