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

"""A celestial body that can compute its sky position, inspired by PyEphem."""

import re

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, ICRS, AltAz, Angle
from astropy.coordinates import solar_system_ephemeris, get_body
from astropy.coordinates import TEME, CartesianDifferential, CartesianRepresentation
from sgp4.api import Satrec, WGS72
from sgp4.model import Satrec as SatrecPython
from sgp4.exporter import export_tle


def to_angle(s, sexagesimal_unit=u.deg):
    """Construct an `Angle` with default units.

    This creates an :class:`~astropy.coordinates.Angle` with the following
    default units:

      - A number is in radians.
      - A decimal string ('123.4') is in degrees.
      - A sexagesimal string ('12:34:56.7') or tuple has `sexagesimal_unit`.

    In addition, bytes are decoded to ASCII strings to normalize user inputs.

    Parameters
    ----------
    s : :class:`~astropy.coordinates.Angle` or equivalent
        Anything accepted by `Angle` and also unitless strings, numbers, tuples
    sexagesimal_unit : :class:`~astropy.units.UnitBase` or str, optional
        The unit applied to sexagesimal strings and tuples

    Returns
    -------
    angle : :class:`~astropy.coordinates.Angle`
        Astropy `Angle`
    """
    try:
        return Angle(s)
    except u.UnitsError:
        # Deal with user input
        if isinstance(s, bytes):
            s = s.decode(encoding='ascii')
        # We now have a number, string or tuple without a unit
        if isinstance(s, str) and ':' in s or isinstance(s, tuple):
            return Angle(s, unit=sexagesimal_unit)
        if isinstance(s, str):
            return Angle(s, unit=u.deg)
        else:
            return Angle(s, unit=u.rad)


class Body:
    """A celestial body that can compute() its sky position.

    This is loosely based on PyEphem's `Body` class. It handles both static
    coordinates fixed in some standard frame and dynamic coordinates that
    are computed on the fly, such as Solar System ephemerides and Earth
    satellites.

    A Body represents a single celestial object with a scalar set of
    coordinates at a given time instant, although the :meth:`compute` method
    may return coordinates for multiple observation times.

    Parameters
    ----------
    name : str
        The name of the body
    """

    def __init__(self, name):
        self.name = name

    @staticmethod
    def _check_location(frame):
        """Check that we have a location for computing AltAz coordinates."""
        if isinstance(frame, AltAz) and frame.location is None:
            raise ValueError('Body needs a location to calculate (az, el) coordinates - '
                             'did you specify an Antenna?')

    @classmethod
    def from_edb(cls, line):
        """Build an appropriate `Body` from a line of an XEphem EDB catalogue.

        Only fixed positions without proper motions and earth satellites have
        been implemented.
        """
        try:
            edb_type = line.split(',')[1][0]
        except (AttributeError, IndexError):
            raise ValueError(f'Failed parsing XEphem EDB line: {line}')
        if edb_type == 'f':
            return FixedBody.from_edb(line)
        elif edb_type == 'E':
            return EarthSatelliteBody.from_edb(line)
        else:
            raise ValueError(f'Unsupported XEphem EDB line: {line}')

    def compute(self, frame, obstime, location):
        """Compute the coordinates of the body in the requested frame.

        Parameters
        ----------
        frame : :class:`~astropy.coordinates.BaseCoordinateFrame` or
                :class:`~astropy.coordinates.SkyCoord`
            The frame to transform this body's coordinates into
        obstime : :class:`~astropy.time.Time`
            The time of observation
        location : :class:`~astropy.coordinates.EarthLocation`
            The location of the observer on the Earth

        Returns
        -------
        coord : :class:`~astropy.coordinates.BaseCoordinateFrame` or
                :class:`~astropy.coordinates.SkyCoord`
            The computed coordinates as a new object
        """
        raise NotImplementedError


class FixedBody(Body):
    """A body with a fixed ICRS position.

    Parameters
    ----------
    name : str
        The name of the celestial body
    coord : :class:`~astropy.coordinates.BaseCoordinateFrame` or
            :class:`~astropy.coordinates.SkyCoord`
        The coordinates of the body
    """

    def __init__(self, name, coord):
        super().__init__(name)
        self.coord = coord

    @classmethod
    def from_edb(cls, line):
        """Construct a `FixedBody` from an XEphem database (EDB) entry."""
        fields = line.split(',')
        name = fields[0]
        # Discard proper motion for now (the part after the |)
        ra = fields[2].split('|')[0]
        dec = fields[3].split('|')[0]
        return cls(name, SkyCoord(ra=Angle(ra, unit=u.hour), dec=Angle(dec, unit=u.deg)))

    def to_edb(self):
        """Create an XEphem database (EDB) entry for fixed body ("f").

        See http://www.clearskyinstitute.com/xephem/xephem.html
        """
        icrs = self.coord.transform_to(ICRS)
        return '{},f,{},{}'.format(self.name, icrs.ra.to_string(sep=':', unit=u.hour),
                                   icrs.dec.to_string(sep=':', unit=u.deg))

    def compute(self, frame, obstime=None, location=None):
        """Compute the coordinates of the body in the requested frame.

        Parameters
        ----------
        frame : :class:`~astropy.coordinates.BaseCoordinateFrame` or
                :class:`~astropy.coordinates.SkyCoord`
            The frame to transform this body's coordinate into
        obstime : :class:`~astropy.time.Time`, optional
            The time of observation
        location : :class:`~astropy.coordinates.EarthLocation`, optional
            The location of the observer on the Earth

        Returns
        -------
        coord : :class:`~astropy.coordinates.BaseCoordinateFrame` or
                :class:`~astropy.coordinates.SkyCoord`
            The computed coordinates as a new object
        """
        Body._check_location(frame)
        # If obstime is array-valued and not contained in the output frame, the transform
        # will return a scalar SkyCoord. Repeat the value to match obstime shape instead.
        if (obstime is not None and not obstime.isscalar
           and 'obstime' not in frame.get_frame_attr_names()):
            coord = self.coord.take(np.zeros_like(obstime, dtype=int))
        else:
            coord = self.coord
        return coord.transform_to(frame)


class SolarSystemBody(Body):
    """A major Solar System body identified by name.

    Parameters
    ----------
    name : str
        The name of the Solar System body
    """

    def __init__(self, name):
        if name.lower() not in solar_system_ephemeris.bodies:  # noqa: E1135
            raise ValueError("Unknown Solar System body '{}' - should be one of {}"
                             .format(name.lower(), solar_system_ephemeris.bodies))
        super().__init__(name)

    def compute(self, frame, obstime, location=None):
        """Determine position of body for given time and location and transform to `frame`."""
        Body._check_location(frame)
        gcrs = get_body(self.name, obstime, location)
        return gcrs.transform_to(frame)


def _edb_to_time(edb_epoch):
    """Construct `Time` object from XEphem EDB epoch string."""
    match = re.match(r'\s*(\d{1,2})/(\d+\.?\d*)/\s*(\d+)', edb_epoch, re.ASCII)
    if not match:
        raise ValueError(f"Epoch string '{edb_epoch}' does not match EDB format 'MM/DD.DD+/YYYY'")
    frac_day, int_day = np.modf(float(match[2]))
    # Convert fractional day to hours, minutes and fractional seconds via Astropy machinery.
    # Add arbitrary integer day to suppress ERFA warnings (will be replaced by actual day next).
    rec = Time(59081.0, frac_day, scale='utc', format='mjd').ymdhms
    rec['year'] = int(match[3])
    rec['month'] = int(match[1])
    rec['day'] = int(int_day)
    return Time(rec, scale='utc')


def _time_to_edb(t, high_precision=False):
    """Construct XEphem EDB epoch string from `Time` object."""
    # The output of this function is undefined if `t` is within a leap second
    if not high_precision:
        # The XEphem startok/endok epochs are also single-precision MJDs
        t = Time(np.float32(t.utc.mjd), format='mjd')
    dt = t.utc.datetime
    second = dt.second + dt.microsecond / 1e6
    minute = dt.minute + second / 60.
    hour = dt.hour + minute / 60.
    day = dt.day + hour / 24.
    if high_precision:
        # See write_E function in libastro's dbfmt.c
        return f'{dt.month:d}/{day:.12g}/{dt.year:d}'
    else:
        # See fs_date function in libastro's formats.c
        return f'{dt.month:2d}/{day:02.6g}/{dt.year:-4d}'


class EarthSatelliteBody(Body):
    """Body orbiting the Earth (besides the Moon, which is a SolarSystemBody).

    Parameters
    ----------
    name : str
        The name of the satellite
    satellite : :class:`sgp4.api.Satrec`
        Underlying SGP4 object doing the work with access to satellite parameters
    orbit_number : int, optional
        Number of revolutions / orbits the satellite has completed at given epoch
        (only for backwards compatibility with EDB format, ignore otherwise)
    """

    def __init__(self, name, satellite, orbit_number=0):
        super().__init__(name)
        self.satellite = satellite
        # XXX We store this because C++ sgp4init doesn't take revnum and Satrec object is read-only
        # This needs to go into the XEphem EDB string, which is still the de facto description
        self.orbit_number = orbit_number

    @property
    def epoch(self):
        """The moment in time when the satellite model is true, as an Astropy `Time`."""
        return Time(self.satellite.jdsatepoch, self.satellite.jdsatepochF, scale='utc', format='jd')

    @classmethod
    def from_tle(cls, name, line1, line2):
        """Build an `EarthSatelliteBody` from a two-line element set (TLE).

        Parameters
        ----------
        name : str
            The name of the satellite
        line1, line2 : str
            The two lines of the TLE
        """
        line1 = line1.strip()
        line2 = line2.strip()
        # Use the Python Satrec to validate the TLE first, since the C++ one has no error checking
        SatrecPython.twoline2rv(line1, line2)
        return cls(name, Satrec.twoline2rv(line1, line2))

    def to_tle(self):
        """Export satellite parameters as a TLE in the form `(line1, line2)`."""
        return export_tle(self.satellite)

    @classmethod
    def from_edb(cls, line):
        """Build an `EarthSatelliteBody` from an XEphem database (EDB) entry."""
        fields = line.split(',')
        name = fields[0]
        edb_epoch = _edb_to_time(fields[2].split('|')[0])
        # The SGP4 epoch is the number of days since 1949 December 31 00:00 UT (= JD 2433281.5)
        # Be careful to preserve full 128-bit resolution to enable round-tripping of descriptions
        sgp4_epoch = Time(edb_epoch.jd1 - 2433281.5, edb_epoch.jd2, format='jd').jd
        (inclination, ra_asc_node, eccentricity, arg_perigee, mean_anomaly,
         mean_motion, orbit_decay, orbit_number, drag_coef) = tuple(float(f) for f in fields[3:])
        sat = Satrec()
        sat.sgp4init(
            WGS72,  # gravity model (TLEs are based on WGS72, therefore it is preferred to WGS84)
            'i',           # 'a' = old AFSPC mode, 'i' = improved mode
            0,             # satnum: Satellite number is not stored by XEphem so pick an unused one
            sgp4_epoch,    # epoch
            drag_coef,     # bstar
            (orbit_decay * u.cycle / u.day ** 2).to_value(u.rad / u.minute ** 2),  # ndot
            0.0,                                                         # nddot (not used by SGP4)
            eccentricity,                                                # ecco
            (arg_perigee * u.deg).to_value(u.rad),                       # argpo
            (inclination * u.deg).to_value(u.rad),                       # inclo
            (mean_anomaly * u.deg).to_value(u.rad),                      # mo
            (mean_motion * u.cycle / u.day).to_value(u.rad / u.minute),  # no_kozai
            (ra_asc_node * u.deg).to_value(u.rad),                       # nodeo
        )
        return cls(name, sat, int(orbit_number))

    def to_edb(self):
        """Create an XEphem database (EDB) entry for Earth satellite ("E").

        See http://www.clearskyinstitute.com/xephem/help/xephem.html#mozTocId468501.

        This attempts to be a faithful copy of the write_E function in
        libastro's dbfmt.c, down to its use of single precision floats.
        """
        sat = self.satellite
        epoch = self.epoch
        # Extract orbital elements in XEphem units, and mostly single-precision.
        # The trailing comments are corresponding XEphem variable names.
        inclination = np.float32((sat.inclo * u.rad).to_value(u.deg))                    # inc
        ra_asc_node = np.float32((sat.nodeo * u.rad).to_value(u.deg))                    # raan
        eccentricity = np.float32(sat.ecco)                                              # e
        arg_perigee = np.float32((sat.argpo * u.rad).to_value(u.deg))                    # ap
        mean_anomaly = np.float32((sat.mo * u.rad).to_value(u.deg))                      # M
        # The mean motion uses double precision due to "sensitive differencing operation"
        mean_motion = (sat.no_kozai * u.rad / u.minute).to_value(u.cycle / u.day)        # n
        orbit_decay = (sat.ndot * u.rad / u.minute ** 2).to_value(u.cycle / u.day ** 2)  # decay
        orbit_decay = np.float32(orbit_decay)
        # XXX Satrec object only accepts revnum via twoline2rv but EDB needs it, so add a backdoor
        orbit_number = sat.revnum if sat.revnum else self.orbit_number                   # orbit
        drag_coef = np.float32(sat.bstar)                                                # drag
        epoch_str = _time_to_edb(epoch, high_precision=True)                             # epoch
        if abs(orbit_decay) > 0:
            # The TLE is considered valid until the satellite period changes by more
            # than 1%, but never for more than 100 days either side of the epoch.
            # The mean motion is revs/day while decay is (revs/day)/day.
            stable_days = np.minimum(0.01 * mean_motion / abs(orbit_decay), 100)
            epoch_start = _time_to_edb(epoch - stable_days)                              # startok
            epoch_end = _time_to_edb(epoch + stable_days)                                # endok
            valid_range = f'|{epoch_start}|{epoch_end}'
        else:
            valid_range = ''
        return (f'{self.name},E,{epoch_str}{valid_range},{inclination:.8g},'
                f'{ra_asc_node:.8g},{eccentricity:.8g},{arg_perigee:.8g},'
                f'{mean_anomaly:.8g},{mean_motion:.12g},{orbit_decay:.8g},'
                f'{orbit_number:d},{drag_coef:.8g}')

    def compute(self, frame, obstime, location=None):
        """Determine position of body at the given time and transform to `frame`."""
        Body._check_location(frame)
        # Propagate the satellite according to SGP4 model (use array version if possible)
        if obstime.shape == ():
            e, r, v = self.satellite.sgp4(obstime.jd1, obstime.jd2)
        else:
            e, r, v = self.satellite.sgp4_array(obstime.jd1.ravel(), obstime.jd2.ravel())
            e = e.reshape(obstime.shape)
            r = r.T.reshape((3,) + obstime.shape)
            v = v.T.reshape((3,) + obstime.shape)
        # Represent the position and velocity in the appropriate TEME frame
        teme_p = CartesianRepresentation(r * u.km)
        teme_v = CartesianDifferential(v * (u.km / u.s))
        teme = TEME(teme_p.with_differentials(teme_v), obstime=obstime)
        # Convert to the desired output frame
        return teme.transform_to(frame)


class StationaryBody(Body):
    """Stationary body with fixed (az, el) coordinates.

    This is a simplified :class:`Body` that is useful to specify targets
    such as zenith and geostationary satellites.

    Parameters
    ----------
    az, el : string or float
        Azimuth and elevation, either in 'D:M:S' string format, or float in rads
    name : string, optional
        The name of the stationary body
    """

    def __init__(self, az, el, name=None):
        self.coord = AltAz(az=to_angle(az), alt=to_angle(el))
        if not name:
            name = "Az: {} El: {}".format(self.coord.az.to_string(sep=':', unit=u.deg),
                                          self.coord.alt.to_string(sep=':', unit=u.deg))
        super().__init__(name)

    def compute(self, frame, obstime, location):
        """Transform (az, el) at given location and time to requested `frame`."""
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
                raise ValueError('StationaryBody needs a location to calculate coordinates - '
                                 'did you specify an Antenna?')
            return altaz.transform_to(frame)


class NullBody(FixedBody):
    """Body with no position, used as a placeholder.

    This body has the expected methods of :class:`Body`, but always returns NaNs
    for all coordinates. It is intended for use as a placeholder when no proper
    target object is available, i.e. as a dummy target.
    """

    def __init__(self):
        super().__init__('Nothing', ICRS(np.nan * u.rad, np.nan * u.rad))
