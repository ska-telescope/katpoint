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

import copy
import datetime

import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import ICRS, SkyCoord, AltAz
from astropy.coordinates import solar_system_ephemeris, get_body
from astropy.coordinates import TEME, CartesianDifferential, CartesianRepresentation

import sgp4.model
import sgp4.earth_gravity
from sgp4.propagation import sgp4init

from .ephem_extra import angle_from_degrees


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

    def writedb(self):
        """ Create an XEphem catalogue entry.

        See http://www.clearskyinstitute.com/xephem/xephem.html
        """
        icrs = self.coord.transform_to(ICRS)
        return '{},f,{},{}'.format(self.name, icrs.ra.to_string(sep=':', unit=u.hour),
                                   icrs.dec.to_string(sep=':', unit=u.deg))


class SolarSystemBody(Body):
    """A major Solar System body identified by name.

    Parameters
    ----------
    name : str
        The name of the Solar System body
    """

    def __init__(self, name):
        if name.lower() not in solar_system_ephemeris.bodies:
            raise ValueError("Unknown Solar System body '{}' - should be one of {}"
                             .format(name.lower(), solar_system_ephemeris.bodies))
        super().__init__(name)

    def compute(self, frame, obstime, location=None):
        """Determine position of body for given time and location and transform to `frame`."""
        Body._check_location(frame)
        gcrs = get_body(self.name, obstime, location)
        return gcrs.transform_to(frame)


class EarthSatelliteBody(Body):
    """Body orbiting the Earth (besides the Moon, which is a SolarSystemBody).

    Parameters
    ----------
    name : str
        The name of the satellite
    """

    def __init__(self, name):
        super().__init__(name)

    def compute(self, frame, obstime, location=None):
        """Determine position of body at the given time and transform to `frame`."""
        Body._check_location(frame)
        # Create an SGP4 satellite object
        self._sat = sgp4.model.Satellite()
        self._sat.whichconst = sgp4.earth_gravity.wgs84
        self._sat.satnum = 1

        # Extract date and time from the epoch
        ep = copy.deepcopy(self._epoch)
        ep.format = 'yday'
        y = int(ep.value[:4])
        d = int(ep.value[5:8])
        h = int(ep.value[9:11])
        m = int(ep.value[12:14])
        s = float(ep.value[15:])
        self._sat.epochyr = y
        self._sat.epochdays = d + (h + (m + s / 60.0) / 60.0) / 24.0
        ep.format = 'jd'
        self._sat.jdsatepoch = ep.value
        self._sat.bstar = self._drag
        self._sat.ndot = self._decay
        self._sat.nddot = self._nddot
        self._sat.inclo = float(self._inc)
        self._sat.nodeo = float(self._raan)
        self._sat.ecco = self._e
        self._sat.argpo = self._ap
        self._sat.mo = self._M
        self._sat.no_kozai = self._n / (24.0 * 60.0) * (2.0 * np.pi)

        # Compute position and velocity
        date = obstime.iso
        yr = int(date[:4])
        mon = int(date[5:7])
        day = int(date[8:10])
        h = int(date[11:13])
        m = int(date[14:16])
        s = float(date[17:])
        sgp4init(sgp4.earth_gravity.wgs84, False, self._sat.satnum,
                 self._sat.jdsatepoch-2433281.5, self._sat.bstar,
                 self._sat.ndot, self._sat.nddot,
                 self._sat.ecco, self._sat.argpo, self._sat.inclo,
                 self._sat.mo, self._sat.no,
                 self._sat.nodeo, self._sat)
        p, v = self._sat.propagate(yr, mon, day, h, m, s)

        teme_p = CartesianRepresentation(p * u.km)
        teme_v = CartesianDifferential(v * u.km / u.s)
        teme = TEME(teme_p.with_differentials(teme_v), obstime=obstime)
        return teme.transform_to(frame)

    def writedb(self):
        """ Create an XEphem catalogue entry.

        See http://www.clearskyinstitute.com/xephem/xephem.html
        """
        dt = self._epoch.iso
        yr = int(dt[:4])
        mon = int(dt[5:7])
        day = int(dt[8:10])
        h = int(dt[11:13])
        m = int(dt[14:16])
        s = float(dt[17:])

        # The epoch field contains 3 dates, the actual epoch and the range
        # of valid dates which xepehm sets to +/- 100 days.
        epoch0 = '{0}/{1:.9}/{2}'.format(mon, day + ((h + (m + s/60.0) / 60.0) / 24.0), yr)
        e = self._epoch + TimeDelta(-100, format='jd')
        dt = str(e)
        yr = int(dt[:4])
        mon = int(dt[5:7])
        day = int(dt[8:10])
        h = int(dt[11:13])
        m = int(dt[14:16])
        s = float(dt[17:])
        epoch1 = '{0}/{1:.6}/{2}'.format(mon, day + ((h + (m + s/60.0) / 60.0) / 24.0), yr)
        e = e + TimeDelta(200, format='jd')
        dt = str(e)
        yr = int(dt[:4])
        mon = int(dt[5:7])
        day = int(dt[8:10])
        h = int(dt[11:13])
        m = int(dt[14:16])
        s = float(dt[17:])
        epoch2 = '{0}/{1:.6}/{2}'.format(mon, day + ((h + (m + s/60.0) / 60.0) / 24.0), yr)

        epoch = '{0}| {1}| {2}'.format(epoch0, epoch1, epoch2)

        return '{0},{1},{2},{3},{4},{5:0.6f},{6:0.2f},{7},{8},{9},{10},{11}'.\
            format(self.name, 'E',
                   epoch,
                   np.rad2deg(float(self._inc)),
                   np.rad2deg(float(self._raan)),
                   self._e,
                   np.rad2deg(float(self._ap)),
                   np.rad2deg(float(self._M)),
                   self._n,
                   self._decay,
                   self._orbit,
                   self._drag)


def _tle_to_float(tle_float):
    """ Convert a TLE formatted float to a float."""
    dash = tle_float.find('-')
    if dash == -1:
        return float(tle_float)
    else:
        return float(tle_float[:dash] + "e-" + tle_float[dash+1:])


def readtle(name, line1, line2):
    """Create an EarthSatelliteBody object from a TLE description of an orbit.

    See https://en.wikipedia.org/wiki/Two-line_element_set

    Parameters
    ----------
    name : str
        Satellite name

    line1 : str
        Line 1 of TLE

    line2 : str
        Line 2 of TLE
    """
    line1 = line1.lstrip()
    line2 = line2.lstrip()
    s = EarthSatelliteBody(name)
    epochyr = int('20' + line1[18:20])
    epochdays = float(line1[20:32])

    # Extract day, hour, min, sec from epochdays
    ed = float(epochdays)
    d = int(ed)
    f = ed - d
    h = int(f * 24.0)
    f = (f * 24.0 - h)
    m = int(f * 60.0)
    sec = (f * 60.0 - m) * 60.0
    date = '{0:04d}:{1:03d}:{2:02d}:{3:02d}:{4:02}'.format(epochyr, d, h, m, sec)

    s._epoch = Time('2000-01-01 00:00:00.000')

    s._epoch = Time(date, format='yday')
    s._epoch.format = 'iso'

    s._inc = np.deg2rad(_tle_to_float(line2[8:16]))
    s._raan = np.deg2rad(_tle_to_float(line2[17:25]))
    s._e = _tle_to_float('0.' + line2[26:33])
    s._ap = np.deg2rad(_tle_to_float(line2[34:42]))
    s._M = np.deg2rad(_tle_to_float(line2[43:51]))
    s._n = _tle_to_float(line2[52:63])
    s._decay = _tle_to_float(line1[33:43])
    s._nddot = _tle_to_float(line1[44:52])
    s._orbit = int(line2[63:68])
    s._drag = _tle_to_float('0.' + line1[53:61].strip())

    return s


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
        self.coord = AltAz(az=angle_from_degrees(az), alt=angle_from_degrees(el))
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
