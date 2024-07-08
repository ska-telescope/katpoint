################################################################################
# Copyright (c) 2009-2021,2023, National Research Foundation (SARAO)
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

"""Antenna object containing sufficient information to point at target, correct delays.

An *antenna* is considered to be a steerable parabolic dish containing multiple
feeds. The :class:`Antenna` object wraps the antenna's location, dish diameter
and other parameters that affect pointing and delay calculations.
"""

import functools
from types import SimpleNamespace

import astropy.units as u
import numpy as np
from astropy.coordinates import CartesianRepresentation, EarthLocation

from .conversion import (
    angle_to_string,
    ecef_to_enu,
    enu_to_ecef,
    lla_to_ecef,
    strip_zeros,
    to_angle,
)
from .delay_model import DelayModel
from .pointing import PointingModel
from .timestamp import Timestamp

# Singleton that identifies default antenna parameters
_DEFAULT = object()

# -------------------------------------------------------------------------------------
# --- CLASS :  Antenna
# -------------------------------------------------------------------------------------


@functools.total_ordering
class Antenna:
    """An antenna that can point at a target.

    This is a wrapper around an Astropy `EarthLocation` that adds a dish
    diameter and other parameters related to pointing and delay calculations.

    It has two variants: a stand-alone single dish, or an antenna that is part
    of an array. The first variant is initialised with the antenna location,
    while the second variant is initialised with the array reference location
    and an ENU (east-north-up) offset for the specific antenna which also
    doubles as the first part of a broader delay model for the antenna.

    Additionally, a diameter, a pointing model and a beamwidth factor may be
    specified. These parameters are collected for convenience, and the pointing
    model is not applied by default when calculating pointing or delays.

    The Antenna object is typically passed around in string form, and is fully
    described by its *description string*, which has the following format::

     name, latitude (D:M:S), longitude (D:M:S), altitude (m), diameter (m),
     east-north-up offset (m) / delay model, pointing model, beamwidth

    A stand-alone dish has the antenna location as lat-lon-alt and the ENU
    offset as an empty string, while an antenna that is part of an array has
    the array reference location as lat-lon-alt and the ENU offset as a
    space-separated string of 3 numbers (followed by any additional delay model
    terms). The pointing model is a space-separated string of model parameters
    (or empty string if there is no pointing model). The beamwidth is a single
    floating-point number.

    Any empty fields at the end of the description string may be omitted, as
    they will be replaced by defaults. The first four fields are required
    (but the name may be an empty string).

    Here are some examples of description strings::

     - Single dish
       'XDM, -25:53:23.0, 27:41:03.0, 1406.1086, 15.0'

     - Simple array antenna
       'FF1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 18.4 -8.7 0.0'

     - Fully-specified antenna
       'FF2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 86.2 25.5 0.0, -0:06:39.6 0, 1.16'

    Parameters
    ----------
    antenna : :class:`~astropy.coordinates.EarthLocation`, str or :class:`Antenna`
        A location on Earth, a full description string or existing Antenna object.
        The parameters in the description string or existing Antenna can still
        be overridden by providing additional parameters after `antenna`.
    name : string, optional
        Name of antenna (may be empty but may not contain commas)
    diameter : :class:`~astropy.units.Quantity` or string or float, optional
        Dish diameter, in metres
    delay_model : :class:`DelayModel` object or equivalent, optional
        Delay model for antenna, either as a direct object, a file-like object
        representing a parameter file, or a string or sequence of float params.
        The first three parameters form an East-North-Up offset from WGS84
        reference position, in metres.
    pointing_model : :class:`PointingModel` object or equivalent, optional
        Pointing model for antenna, either as a direct object, a file-like
        object representing a parameter file, or a string or sequence of
        float parameters from which the :class:`PointingModel` object can
        be instantiated
    beamwidth : string or float, optional
        Full width at half maximum (FWHM) average beamwidth, as a multiple of
        lambda / D (wavelength / dish diameter). This depends on the dish
        illumination pattern, and ranges from 1.03 for a uniformly illuminated
        circular dish to 1.22 for a Gaussian-tapered circular dish (the
        default).

    Attributes
    ----------
    location :class:`~astropy.coordinates.EarthLocation`
        Underlying object used for pointing calculations
    ref_location :class:`~astropy.coordinates.EarthLocation`
        Array reference location for antenna in an array (same as
        `location` for a stand-alone antenna)

    Raises
    ------
    ValueError
        If description string has wrong format or parameters are incorrect
    """

    def __init__(
        self,
        antenna,
        name=_DEFAULT,
        diameter=_DEFAULT,
        delay_model=_DEFAULT,
        pointing_model=_DEFAULT,
        beamwidth=_DEFAULT,
    ):
        default = SimpleNamespace(
            name="", diameter=0.0, delay_model=None, pointing_model=None, beamwidth=1.22
        )
        if isinstance(antenna, str):
            # Create a temporary Antenna object to serve up default parameters instead
            antenna = Antenna.from_description(antenna)
        if isinstance(antenna, Antenna):
            default = antenna
            antenna = default.ref_location

        name = default.name if name is _DEFAULT else name
        diameter = default.diameter if diameter is _DEFAULT else diameter
        delay_model = default.delay_model if delay_model is _DEFAULT else delay_model
        pointing_model = (
            default.pointing_model if pointing_model is _DEFAULT else pointing_model
        )
        beamwidth = default.beamwidth if beamwidth is _DEFAULT else beamwidth

        if "," in name:
            raise ValueError(f"Antenna name '{name}' may not contain commas")
        self.name = name
        self.diameter = diameter << u.m
        self.delay_model = DelayModel(delay_model)
        self.pointing_model = PointingModel(pointing_model)
        self.beamwidth = float(beamwidth)
        self.ref_location = self.location = antenna
        if self.delay_model:
            # Convert ENU offset to ECEF coordinates of antenna
            xyz = enu_to_ecef(*self.ref_position_wgs84, *self.position_enu)
            self.location = EarthLocation.from_geocentric(*xyz, unit=u.m)

    def __str__(self):
        """Complete string representation of antenna object."""
        return self.description

    def __repr__(self):
        """Short human-friendly string representation of antenna object."""
        return f"<katpoint.Antenna {self.name!r} diam={self.diameter} at {id(self):#x}>"

    def __reduce__(self):
        """Pickle object based on description string."""
        return (self.__class__, (self.description,))

    def __eq__(self, other):
        """Equality comparison operator."""
        return self.description == (
            other.description if isinstance(other, Antenna) else other
        )

    def __lt__(self, other):
        """Less-than comparison operator (needed for sorting and np.unique)."""
        return self.description < (
            other.description if isinstance(other, Antenna) else other
        )

    def __hash__(self):
        """Compute hash on description string, just like equality operator."""
        return hash(self.description)

    @property
    def ref_position_wgs84(self):
        """WGS84 reference position.

        The latitude and longitude are in radians, and the altitude in metres.
        """
        lon, lat, height = self.ref_location.to_geodetic(ellipsoid="WGS84")
        return (lat.rad, lon.rad, height.to_value(u.m))

    @property
    def position_wgs84(self):
        """WGS84 position.

        The latitude and longitude are in radians, and the altitude in metres.
        """
        lon, lat, height = self.location.to_geodetic(ellipsoid="WGS84")
        return (lat.rad, lon.rad, height.to_value(u.m))

    @property
    def position_enu(self):
        """East-North-Up offset from WGS84 reference position, in metres."""
        dm = self.delay_model
        return (dm["POS_E"], dm["POS_N"], dm["POS_U"])

    @property
    def position_ecef(self):
        """ECEF (Earth-centred Earth-fixed) position of antenna (in metres)."""
        return tuple(self.location.itrs.cartesian.xyz.to_value(u.m))

    @property
    def description(self):
        """Complete string representation of antenna object."""
        # These fields are used to build up the antenna description string
        fields = [self.name]
        # Store `EarthLocation` as WGS84 coordinates
        lon, lat, height = self.ref_location.to_geodetic(ellipsoid="WGS84")
        fields += [
            angle_to_string(lat),
            angle_to_string(lon),
        ]  # these are already in degrees
        # State height to nearest micron (way overkill) to get rid of numerical fluff,
        # using poor man's {:.6g} that avoids scientific notation for very small heights
        fields += [strip_zeros(f"{height.to_value(u.m):.6f}")]
        fields += [strip_zeros(f"{self.diameter.to_value(u.m):.6f}")]
        fields += [self.delay_model.description]
        fields += [self.pointing_model.description]
        fields += [str(self.beamwidth)]
        return ", ".join(fields)

    @classmethod
    def from_description(cls, description):
        """Construct antenna object from description string."""
        errmsg_prefix = f"Antenna description string '{description}' "
        if not description:
            raise ValueError(errmsg_prefix + "is empty")
        try:
            description.encode("ascii")
        except UnicodeError as err:
            raise ValueError(errmsg_prefix + "contains non-ASCII characters") from err
        # Split description string on commas
        fields = [s.strip() for s in description.split(",")]
        # Extract required fields
        if len(fields) < 4:
            raise ValueError(errmsg_prefix + "has fewer than four fields")
        name, latitude, longitude, altitude = fields[:4]
        # Construct Earth location from WGS84 coordinates
        location = EarthLocation(
            lat=to_angle(latitude), lon=to_angle(longitude), height=altitude
        )
        return cls(location, name, *fields[4:8])

    def baseline_toward(self, antenna2):
        """Baseline vector pointing toward second antenna, in ENU coordinates.

        This calculates the baseline vector pointing from this antenna toward a
        second antenna, *antenna2*, in local East-North-Up (ENU) coordinates
        relative to this antenna's geodetic location.

        Parameters
        ----------
        antenna2 : :class:`~astropy.coordinates.EarthLocation` or :class:`Antenna`
            Second antenna of baseline pair (baseline vector points toward it)

        Returns
        -------
        enu : :class:`~astropy.coordinates.CartesianRepresentation`
            East, North, Up coordinates of baseline vector as Cartesian (x, y, z)
        """
        antenna2 = Antenna(antenna2)
        # If this antenna is at reference position of second antenna,
        # simply return its ENU offset
        if np.array_equal(self.position_wgs84, antenna2.ref_position_wgs84):
            enu = antenna2.position_enu
        else:
            enu = ecef_to_enu(
                *self.position_wgs84, *lla_to_ecef(*antenna2.position_wgs84)
            )
        return CartesianRepresentation(*enu, unit=u.m)

    def local_sidereal_time(self, timestamp=None):
        """Calculate local apparent sidereal time at antenna for timestamp(s).

        This is a vectorised function that returns the local apparent sidereal
        time at the antenna for the given timestamp(s).

        Parameters
        ----------
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equiv, optional
            Timestamp(s), defaults to now

        Returns
        -------
        last : :class:`astropy.coordinates.Longitude`
            Local apparent sidereal time(s)
        """
        time = Timestamp(timestamp).time
        return time.sidereal_time("apparent", longitude=self.location.lon)

    def array_reference_antenna(self, name="array"):
        """Synthetic antenna at the delay model reference position of this antenna.

        This is mainly useful as the reference `antenna` for
        :meth:`.Target.uvw`, in which case it will give both faster and more
        accurate results than other choices.

        The returned antenna will have no delay or pointing model. It is
        intended to be used only for its position and does not correspond to a
        physical antenna.
        """
        return Antenna(self.ref_location, name)
