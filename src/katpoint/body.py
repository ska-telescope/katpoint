################################################################################
# Copyright (c) 2019-2023, National Research Foundation (SARAO)
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

"""A celestial body that can compute its sky position, inspired by PyEphem."""

import re

import astropy.units as u
import numpy as np
from astropy.coordinates import (
    GCRS,
    ICRS,
    TEME,
    AltAz,
    Angle,
    CartesianDifferential,
    CartesianRepresentation,
    SkyCoord,
    UnitSphericalRepresentation,
    get_body,
    solar_system_ephemeris,
)
from astropy.time import Time
from sgp4.api import WGS72, Satrec
from sgp4.exporter import export_tle
from sgp4.model import Satrec as SatrecPython

from .conversion import angle_to_string, to_angle


class Body:
    """A celestial body that can compute() its sky position.

    This is loosely based on PyEphem's `Body` class. It handles both static
    coordinates fixed in some standard frame and dynamic coordinates that
    are computed on the fly, such as Solar System ephemerides and Earth
    satellites.

    A Body represents a single celestial object with a scalar set of
    coordinates at a given time instant, although the :meth:`compute` method
    may return coordinates for multiple observation times.
    """

    @property
    def default_name(self):
        """A default name for the body derived from its coordinates or properties."""
        return "Unknown"

    @property
    def tag(self):
        """The type of body, as a string tag."""
        return "unknown"

    def __repr__(self):
        """Short human-friendly string representation of target object."""
        class_name = "katpoint.body." + self.__class__.__name__
        return f"<{class_name} {self.default_name!r} at {id(self):#x}>"

    @staticmethod
    def _check_location(frame):
        """Check that we have a location for computing AltAz coordinates."""
        if isinstance(frame, AltAz) and frame.location is None:
            raise ValueError(
                "Body needs a location to calculate (az, el) coordinates - "
                "did you specify an Antenna?"
            )

    @classmethod
    def from_edb(cls, line):
        """Build an appropriate `Body` from a line of an XEphem EDB catalogue.

        Only fixed positions without proper motions and earth satellites have
        been implemented.
        """
        try:
            edb_type = line.split(",")[1][0]
        except (AttributeError, IndexError) as err:
            raise ValueError(f"Failed parsing XEphem EDB line: {line}") from err
        if edb_type == "f":
            return FixedBody.from_edb(line)
        elif edb_type == "E":
            return EarthSatelliteBody.from_edb(line)
        else:
            raise ValueError(f"Unsupported XEphem EDB line: {line}")

    def compute(self, frame, obstime, location=None, to_celestial_sphere=False):
        """Compute the coordinates of the body in the requested frame.

        Parameters
        ----------
        frame : :class:`~astropy.coordinates.BaseCoordinateFrame` or
                :class:`~astropy.coordinates.SkyCoord`
            The frame to transform this body's coordinates into
        obstime : :class:`~astropy.time.Time`
            The time of observation
        location : :class:`~astropy.coordinates.EarthLocation`, optional
            The location of the observer on the Earth. If not provided,
            take it from `frame` or the body itself, or fall back to the
            centre of the Earth if possible.
        to_celestial_sphere : bool, optional
            Project the body onto the topocentric celestial sphere
            before transforming to `frame`. This is useful to get
            astrometric (ra, dec), for instance.

        Returns
        -------
        coord : :class:`~astropy.coordinates.BaseCoordinateFrame` or
                :class:`~astropy.coordinates.SkyCoord`
            The computed coordinates as a new object

        Raises
        ------
        ValueError
            If `location` is needed but not provided (e.g. for AltAz)
        """
        raise NotImplementedError


def _to_celestial_sphere(coord, obstime, location=None):
    """Turn `coord` into GCRS direction vector relative to `location` at `obstime`."""
    # Use `location`, `coord.location` or the centre of the Earth (i.e. geocentric GCRS)
    if location is None:
        location = getattr(coord, "location", None)
    if location is None:
        obsgeoloc, obsgeovel = None, None
    else:
        obsgeoloc, obsgeovel = location.get_gcrs_posvel(obstime)
    # Get to topocentric GCRS
    gcrs = coord.transform_to(
        GCRS(obstime=obstime, obsgeoloc=obsgeoloc, obsgeovel=obsgeovel)
    )
    # Discard distance from observer as well as any differentials
    # (needed for EarthSatelliteBody)
    return gcrs.realize_frame(gcrs.represent_as(UnitSphericalRepresentation, s=None))


class FixedBody(Body):
    """A body with a fixed ICRS position.

    Parameters
    ----------
    coord : :class:`~astropy.coordinates.BaseCoordinateFrame` or
            :class:`~astropy.coordinates.SkyCoord`
        The coordinates of the body
    """

    def __init__(self, coord):
        self.coord = coord

    @property
    def default_name(self):
        """A default name for the body derived from its coordinates or properties."""
        ra = angle_to_string(self.coord.ra, unit=u.hour, show_unit=False)
        dec = angle_to_string(self.coord.dec, unit=u.deg, show_unit=False)
        return f"Ra: {ra} Dec: {dec}"

    @property
    def tag(self):
        """The type of body, as a string tag."""
        return "radec"

    @classmethod
    def from_edb(cls, line):
        """Construct a `FixedBody` from an XEphem database (EDB) entry."""
        fields = line.split(",")
        # Discard proper motion for now (the part after the |)
        ra = fields[2].split("|")[0]
        dec = fields[3].split("|")[0]
        return cls(SkyCoord(ra=Angle(ra, unit=u.hour), dec=Angle(dec, unit=u.deg)))

    def to_edb(self, name=""):
        """Create an XEphem database (EDB) entry for fixed body ("f").

        See http://www.clearskyinstitute.com/xephem/xephem.html
        """
        icrs = self.coord.transform_to(ICRS())
        ra = angle_to_string(icrs.ra, unit=u.hour, show_unit=False)
        dec = angle_to_string(icrs.dec, unit=u.deg, show_unit=False)
        return f"{name},f,{ra},{dec}"

    def compute(self, frame, obstime, location=None, to_celestial_sphere=False):
        """Compute the coordinates of the fixed body in the requested frame."""
        Body._check_location(frame)
        # If obstime is array-valued and not contained in the output frame,
        # the transform will return a scalar SkyCoord. Repeat the value to
        # match obstime shape instead.
        if (
            obstime is not None
            and not obstime.isscalar
            and "obstime" not in frame.frame_attributes
        ):
            coord = self.coord.take(np.zeros_like(obstime, dtype=int))
        else:
            coord = self.coord
        # If coordinate is dimensionless, it is already on the celestial sphere
        is_unitspherical = (
            isinstance(coord.data, UnitSphericalRepresentation)
            or coord.cartesian.x.unit == u.one
        )
        if to_celestial_sphere and not is_unitspherical:
            coord = _to_celestial_sphere(coord, obstime, location)
        return coord.transform_to(frame)


class GalacticBody(FixedBody):
    """A body with a fixed Galactic position."""

    @property
    def default_name(self):
        """A default name for the body derived from its coordinates or properties."""
        gal_l = angle_to_string(self.coord.l, unit=u.deg, decimal=True, show_unit=False)
        gal_b = angle_to_string(self.coord.b, unit=u.deg, decimal=True, show_unit=False)
        return f"Galactic l: {gal_l} b: {gal_b}"

    @property
    def tag(self):
        """The type of body, as a string tag."""
        return "gal"


class SolarSystemBody(Body):
    """A major Solar System body identified by name.

    Parameters
    ----------
    name : str
        The name of the Solar System body
    """

    def __init__(self, name):
        # pylint: disable=unsupported-membership-test
        if name.lower() not in solar_system_ephemeris.bodies:
            raise ValueError(
                f"Unknown Solar System body '{name.lower()}' - "
                f"should be one of {solar_system_ephemeris.bodies}"
            )
        self._name = name

    @property
    def default_name(self):
        """A default name for the body derived from its coordinates or properties."""
        return self._name

    @property
    def tag(self):
        """The type of body, as a string tag."""
        return "special"

    def compute(self, frame, obstime, location=None, to_celestial_sphere=False):
        """Determine body position for given time + location, transform to `frame`."""
        Body._check_location(frame)
        gcrs = get_body(self._name, obstime, location)
        if to_celestial_sphere:
            # Discard distance from observer
            gcrs = gcrs.realize_frame(gcrs.represent_as(UnitSphericalRepresentation))
        return gcrs.transform_to(frame)


def _edb_to_time(edb_epoch):
    """Construct `Time` object from XEphem EDB epoch string."""
    match = re.match(r"\s*(\d{1,2})/(\d+\.?\d*)/\s*(\d+)", edb_epoch, re.ASCII)
    if not match:
        raise ValueError(
            f"Epoch string '{edb_epoch}' does not match EDB format 'MM/DD.DD+/YYYY'"
        )
    frac_day, int_day = np.modf(float(match[2]))
    # Convert fractional day to hours, minutes and fractional seconds
    # via Astropy machinery. Add arbitrary integer day to suppress
    # ERFA warnings (will be replaced by actual day next).
    rec = Time(59081.0, frac_day, scale="utc", format="mjd").ymdhms
    rec["year"] = int(match[3])
    rec["month"] = int(match[1])
    rec["day"] = int(int_day)
    return Time(rec, scale="utc")


def _time_to_edb(t, high_precision=False):
    """Construct XEphem EDB epoch string from `Time` object."""
    # The output of this function is undefined if `t` is within a leap second
    if not high_precision:
        # The XEphem startok/endok epochs are also single-precision MJDs
        t = Time(np.float32(t.utc.mjd), format="mjd")
    dt = t.utc.datetime
    second = dt.second + dt.microsecond / 1e6
    minute = dt.minute + second / 60.0
    hour = dt.hour + minute / 60.0
    day = dt.day + hour / 24.0
    if high_precision:
        # See write_E function in libastro's dbfmt.c
        return f"{dt.month:d}/{day:.12g}/{dt.year:d}"
    else:
        # See fs_date function in libastro's formats.c
        return f"{dt.month:2d}/{day:02.6g}/{dt.year:-4d}"


class EarthSatelliteBody(Body):
    """Body orbiting the Earth (besides the Moon, which is a SolarSystemBody).

    Parameters
    ----------
    satellite : :class:`sgp4.api.Satrec`
        Underlying SGP4 object doing the work with access to satellite parameters
    orbit_number : int, optional
        Number of revolutions / orbits the satellite has completed at given epoch
        (only for backwards compatibility with EDB format, ignore otherwise)
    """

    def __init__(self, satellite, orbit_number=0):
        self.satellite = satellite
        # XXX We store this because C++ sgp4init doesn't take revnum and Satrec
        # object is read-only. This needs to go into the XEphem EDB string,
        # which is still the de facto description.
        self.orbit_number = orbit_number

    @property
    def default_name(self):
        """A default name for the body derived from its coordinates or properties."""
        # Identify the satellite with its orbit. The orbit can be approximated
        # by 3 numbers if it is nearly circular (2 orientation angles and 1 size).
        # We don't care about the epoch or anomaly which describe the current
        # location along the orbit. The TLE already contains these numbers as
        # strings in convenient units. :-)
        _, line2 = self.to_tle()
        inclination = line2[8:16].strip()
        ra_asc_node = line2[17:25].strip()
        mean_motion = line2[52:63].strip()
        return f"Inc: {inclination} Raan: {ra_asc_node} Rev/day: {mean_motion}"

    @property
    def tag(self):
        """The type of body, as a string tag."""
        # The EDB format only stores essential elements so no NORAD SATCAT ID
        # -> detect it that way
        return "tle" if self.satellite.satnum else "xephem tle"

    @property
    def epoch(self):
        """The moment in time when the satellite model is true, as an Astropy `Time`."""
        return Time(
            self.satellite.jdsatepoch,
            self.satellite.jdsatepochF,
            scale="utc",
            format="jd",
        )

    @classmethod
    def from_tle(cls, line1, line2):
        """Build an `EarthSatelliteBody` from a two-line element set (TLE).

        Parameters
        ----------
        line1, line2 : str
            The two lines of the TLE
        """
        line1 = line1.strip()
        line2 = line2.strip()
        # Use Python Satrec to validate TLE first, since C++ one has no error checking
        SatrecPython.twoline2rv(line1, line2)
        return cls(Satrec.twoline2rv(line1, line2))

    def to_tle(self):
        """Export satellite parameters as a TLE in the form `(line1, line2)`."""
        return export_tle(self.satellite)

    @classmethod
    def from_edb(cls, line):
        """Build an `EarthSatelliteBody` from an XEphem database (EDB) entry."""
        fields = line.split(",")
        edb_epoch = _edb_to_time(fields[2].split("|")[0])
        # The SGP4 epoch is the number of days since 1949 December 31 00:00 UT
        # (= JD 2433281.5). Be careful to preserve full 128-bit resolution
        # to enable round-tripping of descriptions.
        sgp4_epoch = Time(edb_epoch.jd1 - 2433281.5, edb_epoch.jd2, format="jd").jd
        (
            inclination,
            ra_asc_node,
            eccentricity,
            arg_perigee,
            mean_anomaly,
            mean_motion,
            orbit_decay,
            orbit_number,
            drag_coef,
        ) = tuple(float(f) for f in fields[3:])
        sat = Satrec()
        sat.sgp4init(
            WGS72,  # gravity model (TLEs are based on WGS72 therefore prefer to WGS84)
            "i",  # 'a' = old AFSPC mode, 'i' = improved mode
            0,  # satnum: Satellite number is not stored by XEphem so pick unused one
            sgp4_epoch,  # epoch
            drag_coef,  # bstar
            (orbit_decay * u.cycle / u.day**2).to_value(u.rad / u.minute**2),  # ndot
            0.0,  # nddot (not used by SGP4)
            eccentricity,  # ecco
            (arg_perigee * u.deg).to_value(u.rad),  # argpo
            (inclination * u.deg).to_value(u.rad),  # inclo
            (mean_anomaly * u.deg).to_value(u.rad),  # mo
            (mean_motion * u.cycle / u.day).to_value(u.rad / u.minute),  # no_kozai
            (ra_asc_node * u.deg).to_value(u.rad),  # nodeo
        )
        return cls(sat, int(orbit_number))

    def to_edb(self, name=""):
        """Create an XEphem database (EDB) entry for Earth satellite ("E").

        See http://www.clearskyinstitute.com/xephem/help/xephem.html#mozTocId468501.

        This attempts to be a faithful copy of the write_E function in
        libastro's dbfmt.c, down to its use of single precision floats.
        """
        sat = self.satellite
        epoch = self.epoch
        # Extract orbital elements in XEphem units, and mostly single-precision.
        # The trailing comments are corresponding XEphem variable names.
        inclination = np.float32((sat.inclo * u.rad).to_value(u.deg))  # inc
        ra_asc_node = np.float32((sat.nodeo * u.rad).to_value(u.deg))  # raan
        eccentricity = np.float32(sat.ecco)  # e
        arg_perigee = np.float32((sat.argpo * u.rad).to_value(u.deg))  # ap
        mean_anomaly = np.float32((sat.mo * u.rad).to_value(u.deg))  # M
        # Mean motion uses double precision due to "sensitive differencing operation"
        mean_motion = (sat.no_kozai * u.rad / u.minute).to_value(u.cycle / u.day)  # n
        orbit_decay = (sat.ndot * u.rad / u.minute**2).to_value(
            u.cycle / u.day**2
        )  # decay
        orbit_decay = np.float32(orbit_decay)
        # XXX Satrec object only accepts revnum via twoline2rv but EDB needs it,
        # so add a backdoor.
        orbit_number = sat.revnum if sat.revnum else self.orbit_number  # orbit
        drag_coef = np.float32(sat.bstar)  # drag
        epoch_str = _time_to_edb(epoch, high_precision=True)  # epoch
        if abs(orbit_decay) > 0:
            # The TLE is considered valid until the satellite period changes by more
            # than 1%, but never for more than 100 days either side of the epoch.
            # The mean motion is revs/day while decay is (revs/day)/day.
            stable_days = np.minimum(0.01 * mean_motion / abs(orbit_decay), 100) * u.day
            epoch_start = _time_to_edb(epoch - stable_days)  # startok
            epoch_end = _time_to_edb(epoch + stable_days)  # endok
            valid_range = f"|{epoch_start}|{epoch_end}"
        else:
            valid_range = ""
        return (
            f"{name},E,{epoch_str}{valid_range},{inclination:.8g},"
            f"{ra_asc_node:.8g},{eccentricity:.8g},{arg_perigee:.8g},"
            f"{mean_anomaly:.8g},{mean_motion:.12g},{orbit_decay:.8g},"
            f"{orbit_number:d},{drag_coef:.8g}"
        )

    def compute(self, frame, obstime, location=None, to_celestial_sphere=False):
        """Determine position of body at the given time and transform to `frame`."""
        Body._check_location(frame)
        # Propagate satellite according to SGP4 model (use array version if possible)
        if obstime.shape == ():
            e, r, v = self.satellite.sgp4(obstime.jd1, obstime.jd2)
        else:
            e, r, v = self.satellite.sgp4_array(
                obstime.jd1.ravel(), obstime.jd2.ravel()
            )
            e = e.reshape(obstime.shape)
            r = r.T.reshape((3,) + obstime.shape)
            v = v.T.reshape((3,) + obstime.shape)
        # Represent the position and velocity in the appropriate TEME frame
        teme_p = CartesianRepresentation(r * u.km)
        teme_v = CartesianDifferential(v * (u.km / u.s))
        satellite = TEME(teme_p.with_differentials(teme_v), obstime=obstime)
        if to_celestial_sphere:
            satellite = _to_celestial_sphere(satellite, obstime, location)
        # Convert to the desired output frame
        return satellite.transform_to(frame)


class StationaryBody(Body):
    """Stationary body with fixed (az, el) coordinates.

    This is a simplified :class:`Body` that is useful to specify targets
    such as zenith and geostationary satellites.

    Parameters
    ----------
    az, el : string or float
        Azimuth and elevation, either in 'D:M:S' string format, or float in rads
    """

    def __init__(self, az, el):
        self.coord = AltAz(az=to_angle(az), alt=to_angle(el))

    @property
    def default_name(self):
        """A default name for the body derived from its coordinates or properties."""
        az = angle_to_string(self.coord.az, unit=u.deg, show_unit=False)
        el = angle_to_string(self.coord.alt, unit=u.deg, show_unit=False)
        return f"Az: {az} El: {el}"

    @property
    def tag(self):
        """The type of body, as a string tag."""
        return "azel"

    def compute(self, frame, obstime, location=None, to_celestial_sphere=False):
        """Transform (az, el) at given location and time to requested `frame`."""
        # Ignore `to_celestial_sphere` setting since we are already on the sphere :-)
        # Ensure that coordinates have same shape as obstime (broadcasting fails)
        altaz = self.coord.take(np.zeros_like(obstime, dtype=int))
        altaz = altaz.replicate(obstime=obstime, location=location)
        # Bypass transform_to for AltAz -> AltAz, otherwise we need a location
        # and the output (az, el) will not be exactly equal to the coord (az, el)
        # due to small numerical errors.
        if isinstance(frame, AltAz) and altaz.is_equivalent_frame(frame):
            return altaz
        else:
            if location is None:
                raise ValueError(
                    "StationaryBody needs a location to calculate coordinates - "
                    "did you specify an Antenna?"
                )
            return altaz.transform_to(frame)


class NullBody(FixedBody):
    """Body with no position, used as a placeholder.

    This body has the expected methods of :class:`Body`, but always returns NaNs
    for all coordinates. It is intended for use as a placeholder when no proper
    target object is available, i.e. as a dummy target.
    """

    def __init__(self):
        super().__init__(ICRS(np.nan * u.rad, np.nan * u.rad))

    @property
    def default_name(self):
        """A default name for the body derived from its coordinates or properties."""
        return "Nothing"

    @property
    def tag(self):
        """The type of body, as a string tag."""
        return "special"
