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

"""Replacement for the pyephem body classes.

We only implement ra, dec (CIRS), a_ra, a_dec (ICRS RA/dec) and alt, az
(topcentric) as that is all that katpoint uses

The real pyephem computes observed place but katpoint always sets the
pressure to zero so we compute apparent places instead.
"""

import copy
import datetime
import numpy as np

from astropy.coordinates import get_moon
from astropy.coordinates import get_body
from astropy.coordinates import get_sun
from astropy.coordinates import EarthLocation
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import CIRS
from astropy.coordinates import ICRS
from astropy.coordinates import AltAz
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.time import TimeDelta
from astropy import coordinates
from astropy import units

import sgp4.model
import sgp4.earth_gravity
from sgp4.propagation import sgp4init

import pyorbital.geoloc
import pyorbital.astronomy

from .ephem_extra import angle_from_degrees

class Body(object):
    """Base class for all Body classes.

    Attributes
    ----------

    a_radec : astropy.coordinates.SkyCoord
        Astrometric (ICRS) position at date of observation

    radec : astropy.coordinates.SkyCoord
        Apparent (CIRS) position at date of observation

    altaz : astropy.coordinates.AltAz
        Topocentric alt/az of body at date of observation
    """
    def __init__(self):
        self._epoch = Time(2000.0, format='jyear')

    def _compute(self, loc, date, pressure, icrs_radec):
        """Calculates the RA/Dec and Az/El of the body.

        Parameters
        ----------
        loc : astropy.coordinates.EarthLocation
            Location of observation

        date : astropy.Time
            Date of observation

        pressure : float
            Atmospheric pressure

        icrs_radec : astropy.coordinates.SkyCoord
            Position of the body in the ICRS
        """

        # Store the astrometric (ICRS) position
        self.a_radec = icrs_radec

        # Store apparent (CIRS) position
        self.radec = self.a_radec.transform_to(CIRS(obstime=date))

        # ICRS to Az/El
        self.altaz = self.radec.transform_to(AltAz(location=loc,
                obstime=date, pressure=pressure))


class FixedBody(Body):
    """A body with a fixed (on the celestial sphere) position.

    Attributes
    ----------

    _radec : astropy.coordinates.SkyCoord
        Position of body in some RA/Dec frame
    """
    def __init__(self):
        Body.__init__(self)

    def compute(self, loc, date, pressure):
        """Compute alt/az of body.

        Parameters
        ----------
        loc : astropy.coordinates.EarthLocation
            Location of observation

        date : astropy.Time
            Date of observation

        pressure : float
            Atmospheric pressure
        """
        icrs = self._radec.transform_to(ICRS)
        Body._compute(self, loc, date, pressure, icrs)

    def writedb(self):
        """ Create an XEphem catalogue entry.

        See http://www.clearskyinstitute.com/xephem/xephem.html
        """
        icrs = self._radec.transform_to(ICRS)
        return '{},f,{},{}'.format(self.name, icrs.ra.to_string(sep=':', unit=units.hour),
                                   icrs.dec.to_string(sep=':', unit=units.deg))


class Sun(Body):
    def __init__(self):
        Body.__init__(self)
        self.name = 'Sun'

    def compute(self, loc, date, pressure):
        sun = get_sun(date)
        icrs = sun.transform_to(ICRS)
        Body._compute(self, loc, date, pressure, icrs)


class Moon(Body):
    def __init__(self):
        Body.__init__(self)
        self.name = 'Moon'

    def compute(self, loc, date, pressure):
        moon = get_moon(date, loc)
        icrs = moon.transform_to(ICRS)
        Body._compute(self, loc, date, pressure, icrs)

class Earth(Body):
    def __init__(self):
        Body.__init__(self)

    def compute(self):
        pass

class Planet(Body):
    def __init__(self, name):
        Body.__init__(self)
        self._name = name

    def compute(self, loc, date, pressure):
        with solar_system_ephemeris.set('builtin'):
            planet = get_body(self._name, date, loc)
        icrs = planet.transform_to(ICRS)
        Body._compute(self, loc, date, pressure, icrs)

class Mercury(Planet):
    def __init__(self):
        Planet.__init__(self, 'mercury')
        self.name = 'Mercury'

class Venus(Planet):
    def __init__(self):
        Planet.__init__(self, 'venus')
        self.name = 'Venus'

class Mars(Planet):
    def __init__(self):
        Planet.__init__(self, 'mars')
        self.name = 'Mars'

class Jupiter(Planet):
    def __init__(self):
        Planet.__init__(self, 'jupiter')
        self.name = 'Jupiter'

class Saturn(Planet):
    def __init__(self):
        Planet.__init__(self, 'saturn')
        self.name = 'Saturn'

class Uranus(Planet):
    def __init__(self):
        Planet.__init__(self, 'uranus')
        self.name = 'Uranus'

class Neptune(Planet):
    def __init__(self):
        Planet.__init__(self, 'neptune')
        self.name = 'Neptune'

class EarthSatellite(Body):
    """Body orbiting the earth."""
    def __init__(self):
        Body.__init__(self)

    def compute(self, loc, date, pressure):

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
        self._sat.no_kozai = self._n / (24.0 *60.0) * (2.0 * np.pi)

        # Compute position and velocity
        date = date.iso
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
        utc_time = datetime.datetime(yr, mon, day, h, m, int(s),
                int(s - int(s)) * 1000000)
        pos = np.array(p)
        lon, lat, alt = pyorbital.geoloc.get_lonlatalt(pos, utc_time)

        # Convert to alt, az at observer
        az, alt = get_observer_look(lon, lat, alt, utc_time,
                loc.lon.deg, loc.lat.deg, loc.height.to(units.kilometer).value)

        self.altaz = SkyCoord(az*units.deg, alt*units.deg, location=loc,
                obstime=date, pressure=pressure, frame=AltAz)
        self.a_radec = self.altaz.transform_to(ICRS)

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
        epoch0 = '{0}/{1:.9}/{2}'.format(mon,
                day + ((h + (m + s/60.0) / 60.0) / 24.0), yr)
        e = self._epoch + TimeDelta(-100, format='jd')
        dt = str(e)
        yr = int(dt[:4])
        mon = int(dt[5:7])
        day = int(dt[8:10])
        h = int(dt[11:13])
        m = int(dt[14:16])
        s = float(dt[17:])
        epoch1 = '{0}/{1:.6}/{2}'.format(mon,
                day + ((h + (m + s/60.0) / 60.0) / 24.0), yr)
        e = e + TimeDelta(200, format='jd')
        dt = str(e)
        yr = int(dt[:4])
        mon = int(dt[5:7])
        day = int(dt[8:10])
        h = int(dt[11:13])
        m = int(dt[14:16])
        s = float(dt[17:])
        epoch2 = '{0}/{1:.6}/{2}'.format(mon,
                day + ((h + (m + s/60.0) / 60.0) / 24.0), yr)

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
    """Create an EarthSatellite object from a TLE description of an orbit.

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
    s = EarthSatellite()
    s.name = name
    epochyr = int('20' + line1[18:20])
    epochdays = float(line1[20:32])

    # Extract day, hour, min, sec from epochdays
    ed = float(epochdays)
    d = int(ed)
    f = ed - d
    h = int(f * 24.0)
    f = (f * 24.0  - h)
    m = int(f * 60.0)
    sec = (f * 60.0 - m) * 60.0
    date = '{0:04d}:{1:03d}:{2:02d}:{3:02d}:{4:02}'.format(epochyr, d, h, m ,
            sec)

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
        self._azel = AltAz(az=angle_from_degrees(az),
                alt=angle_from_degrees(el))
        if not name:
            name = "Az: {} El: {}".format(self._azel.az.to_string(sep=':', unit=units.deg),
                                          self._azel.alt.to_string(sep=':', unit=units.deg))
        self.name = name

    def compute(self, loc, date, pressure):
        """Update target coordinates for given observer.

        This updates the (ra, dec) coordinates of the target, as seen from the
        given *observer*, while its (az, el) coordinates remain unchanged.

        """
        altaz = AltAz(az=self._azel.az, alt=self._azel.alt, location=loc,
                obstime=date, pressure=pressure)
        icrs = altaz.transform_to(ICRS)
        Body._compute(self, loc, date, pressure, icrs)


class NullBody(object):
    """Body with no position, used as a placeholder.

    This body has the expected methods of :class:`Body`, but always returns NaNs
    for all coordinates. It is intended for use as a placeholder when no proper
    target object is available, i.e. as a dummy target.

    """
    def __init__(self):
        self.name = 'Nothing'
        self.altaz = None
        self.a_radec = None
        self.radec = None

    def compute(self, loc, date, pressure):
        pass
