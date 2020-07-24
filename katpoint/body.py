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

import sgp4.model
import sgp4.earth_gravity
from sgp4.propagation import sgp4init

import pyorbital.geoloc
import pyorbital.astronomy

from .ephem_extra import angle_from_degrees


class Body:
    """A celestial body that can compute() its sky position.

    This is loosely based on PyEphem's `Body` class. It handles both static
    coordinates fixed in some standard frame and dynamic coordinates that
    are computed on the fly, such as Solar System ephemerides and Earth
    satellites.

    Parameters
    ----------
    name : str
        Name of celestial body
    coord : :class:`~astropy.coordinates.BaseCoordinateFrame` or
            :class:`~astropy.coordinates.SkyCoord`, optional
        Coordinates of body (None if it is not fixed in any standard frame)
    """

    def __init__(self, name, coord=None):
        self.name = name
        self.coord = coord

    def compute(self, frame, obstime=None, location=None):
        """Compute the coordinates of the body in the requested frame.

        Parameters
        ----------
        frame : str, :class:`~astropy.coordinates.BaseCoordinateFrame` class or
                instance, or :class:`~astropy.coordinates.SkyCoord` instance
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
        return self.coord.transform_to(frame)


class FixedBody(Body):
    """A body with a fixed position on the celestial sphere."""

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
    name : str or other
        The name of the body (see :func:``~astropy.coordinates.get_body`
        for more details).
    """

    def __init__(self, name):
        if name.lower() not in solar_system_ephemeris.bodies:
            raise ValueError("Unknown Solar System body '{}' - should be one of {}"
                             .format(name.lower(), solar_system_ephemeris.bodies))
        super().__init__(name, None)

    def compute(self, frame, obstime, location=None):
        """Determine position of body in GCRS at given time and transform to `frame`."""
        gcrs = get_body(self.name, obstime, location)
        return gcrs.transform_to(frame)


class EarthSatelliteBody(Body):
    """Body orbiting the Earth (besides the Moon, which is a SolarSystemBody).

    Parameters
    ----------
    name : str
        Name of body
    """

    def __init__(self, name):
        super().__init__(name, None)

    def compute(self, frame, obstime, location):
        """Determine position of body at the given time and transform to `frame`."""
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

        # Convert to lon/lat/alt
        utc_time = datetime.datetime(yr, mon, day, h, m, int(s), int(s - int(s)) * 1000000)
        pos = np.array(p)
        lon, lat, alt = pyorbital.geoloc.get_lonlatalt(pos, utc_time)

        # Convert to alt, az at observer
        az, alt = get_observer_look(lon, lat, alt, utc_time,
                                    location.lon.deg, location.lat.deg,
                                    location.height.to(u.kilometer).value)

        altaz = SkyCoord(az*u.deg, alt*u.deg, frame=AltAz, obstime=obstime, location=location)
        return altaz.transform_to(frame)

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


def get_observer_look(sat_lon, sat_lat, sat_alt, utc_time, lon, lat, alt):
    """Calculate observers look angle to a satellite.

    http://celestrak.com/columns/v02n02/

    Parameters
    ----------
    utc_time: datetime.datetime
        Observation time

    lon: float
        Longitude of observer position on ground in degrees east

    lat: float
        Latitude of observer position on ground in degrees north

    alt: float
        Altitude above sea-level (geoid) of observer position on ground in km

    Return: (Azimuth, Elevation)
    """
    (pos_x, pos_y, pos_z), (vel_x, vel_y, vel_z) = \
        pyorbital.astronomy.observer_position(
        utc_time, sat_lon, sat_lat, sat_alt)

    (opos_x, opos_y, opos_z), (ovel_x, ovel_y, ovel_z) = \
        pyorbital.astronomy.observer_position(utc_time, lon, lat, alt)

    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)

    theta = (pyorbital.astronomy.gmst(utc_time) + lon) % (2 * np.pi)

    rx = pos_x - opos_x
    ry = pos_y - opos_y
    rz = pos_z - opos_z

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    top_s = sin_lat * cos_theta * rx + \
        sin_lat * sin_theta * ry - cos_lat * rz
    top_e = -sin_theta * rx + cos_theta * ry
    top_z = cos_lat * cos_theta * rx + \
        cos_lat * sin_theta * ry + sin_lat * rz

    az_ = np.arctan(-top_e / top_s)

    az_ = np.where(top_s > 0, az_ + np.pi, az_)
    az_ = np.where(az_ < 0, az_ + 2 * np.pi, az_)

    rg_ = np.sqrt(rx * rx + ry * ry + rz * rz)
    el_ = np.arcsin(top_z / rg_)

    return np.rad2deg(az_), np.rad2deg(el_)


class StationaryBody(Body):
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
        super().__init__(name, AltAz(az=angle_from_degrees(az), alt=angle_from_degrees(el)))
        if not self.name:
            self.name = "Az: {} El: {}".format(self.coord.az.to_string(sep=':', unit=u.deg),
                                               self.coord.alt.to_string(sep=':', unit=u.deg))

    def compute(self, frame, obstime, location):
        """Transform (az, el) at given location and time to requested `frame`."""
        altaz = self.coord.replicate(obstime=obstime, location=location)
        if isinstance(frame, AltAz) and altaz.is_equivalent_frame(frame):
            return altaz
        else:
            return altaz.transform_to(frame)


class NullBody(Body):
    """Body with no position, used as a placeholder.

    This body has the expected methods of :class:`Body`, but always returns NaNs
    for all coordinates. It is intended for use as a placeholder when no proper
    target object is available, i.e. as a dummy target.
    """

    def __init__(self):
        super().__init__('Nothing', ICRS(np.nan * u.rad, np.nan * u.rad))
