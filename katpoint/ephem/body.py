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

import sgp4.io
import sgp4.earth_gravity
from sgp4.propagation import sgp4init

import pyorbital.geoloc
import pyorbital.astronomy

from .constants import J2000
from .angle import astropy_angle
from .angle import hours
from .angle import degrees
from .angle import Angle

class Body(object):
    def __init__(self):
        self._epoch = J2000

    def _compute(self, obs, icrs):

        # Earth location
        loc = EarthLocation(lon=obs._lon.astropy_angle,
                lat=obs._lat.astropy_angle, height=obs.elevation)

        # ICRS
        self.a_ra = astropy_angle(icrs.ra, 'h')
        self.a_dec = astropy_angle(icrs.dec)

        # ICRS to CIRS
        appt = icrs.transform_to(CIRS(obstime=obs.date))
        self.ra = astropy_angle(appt.ra, 'h')
        self.dec = astropy_angle(appt.dec)

        # ICRS to Az/El
        altaz = icrs.transform_to(AltAz(location=loc,
                obstime=obs.date, pressure=obs.pressure))
        self.az = astropy_angle(altaz.az)
        self.alt = astropy_angle(altaz.alt)


class FixedBody(Body):
    def __init__(self):
        Body.__init__(self)

    def compute(self, obs):
        icrs = SkyCoord(ra=self._ra.astropy_angle,
                dec=self._dec.astropy_angle, frame='icrs')
        Body._compute(self, obs, icrs)

    def writedb(self):
        """ Create an XEphem catalogue entry.

        See http://www.clearskyinstitute.com/xephem/xephem.html
        """
        return '{0},f,{1},{2}'.format(self.name, self._ra, self._dec)

class Sun(Body):
    def __init__(self):
        Body.__init__(self)
        self.name = 'Sun'

    def compute(self, obs):
        loc = EarthLocation(lon=obs._lon.astropy_angle,
                lat=obs._lat.astropy_angle, height=obs.elevation)
        moon = get_sun(obs.date)
        icrs = moon.transform_to(ICRS)
        Body._compute(self, obs, icrs)


class Moon(Body):
    def __init__(self):
        Body.__init__(self)
        self.name = 'Moon'

    def compute(self, obs):
        loc = EarthLocation(lon=obs._lon.astropy_angle,
                lat=obs._lat.astropy_angle, height=obs.elevation)
        moon = get_moon(obs.date, loc)
        icrs = moon.transform_to(ICRS)
        Body._compute(self, obs, icrs)

class Earth(Body):
    def __init__(self):
        Body.__init__(self)

    def compute(self):
        pass

class Planet(Body):
    def __init__(self, name):
        Body.__init__(self)
        self._name = name

    def compute(self, obs):
        loc = EarthLocation(lon=obs._lon.astropy_angle,
                lat=obs._lat.astropy_angle, height=obs.elevation)
        with solar_system_ephemeris.set('builtin'):
            planet = get_body(self._name, obs.date, loc)
        icrs = planet.transform_to(ICRS)
        Body._compute(self, obs, icrs)

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
    def __init__(self):
        Body.__init__(self)

    def compute(self, obs):

        # Create an SGP4 satellite object
        self._sat = sgp4.io.Satellite()
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
        self._sat.inclo = float(self._inc)
        self._sat.nodeo = float(self._raan)
        self._sat.ecco = self._e
        self._sat.argpo = self._ap
        self._sat.mo = self._M
        self._sat.no = self._n / (24.0 *60.0) * (2.0 * np.pi)

        # Compute position and velocity
        date = obs.date.iso
        yr = int(date[:4])
        mon = int(date[5:7])
        day = int(date[8:10])
        h = int(date[11:13])
        m = int(date[14:16])
        s = float(date[17:])
        sgp4init(sgp4.earth_gravity.wgs84, False, self._sat.satnum,
                self._sat.jdsatepoch-2433281.5, self._sat.bstar,
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
                np.rad2deg(obs.lon), np.rad2deg(obs.lat), obs.elevation / 1000)

        self.az = degrees(np.deg2rad(az))
        self.alt = degrees(np.deg2rad(alt))
        self.a_ra, self.a_dec = obs.radec_of(self.az, self.alt)

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
    """ Convert a TLE formatted float to a float
    """
    dash = tle_float.find('-')
    if dash == -1:
        return float(tle_float)
    else:
        return float(tle_float[:dash] + "e-" + tle_float[dash+1:])

def readtle(name, line1, line2):
    """ Create an EarthSatellite from a two line element description of
    an orbit.

    See https://en.wikipedia.org/wiki/Two-line_element_set
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

    s._inc = degrees(np.deg2rad(_tle_to_float(line2[8:16])))
    s._raan = degrees(np.deg2rad(_tle_to_float(line2[17:25])))
    s._e = _tle_to_float('0.' + line2[26:33])
    s._ap = degrees(np.deg2rad(_tle_to_float(line2[34:42])))
    s._M = degrees(np.deg2rad(_tle_to_float(line2[43:51])))
    s._n = _tle_to_float(line2[52:63])
    s._decay = _tle_to_float(line1[33:43])
    s._orbit = int(line2[63:68])
    s._drag = _tle_to_float('0.' + line1[53:61].strip())

    return s

def get_observer_look(sat_lon, sat_lat, sat_alt, utc_time, lon, lat, alt):
    """Calculate observers look angle to a satellite.
    http://celestrak.com/columns/v02n02/

    utc_time: Observation time (datetime object)
    lon: Longitude of observer position on ground in degrees east
    lat: Latitude of observer position on ground in degrees north
    alt: Altitude above sea-level (geoid) of observer position on ground in km

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


