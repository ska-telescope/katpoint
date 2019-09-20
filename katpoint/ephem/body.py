"""Replacement for the pyephem body classes.

We only implement ra, dec (CIRS), a_ra, a_dec (ICRS RA/dec) and alt, az
(topcentric) as that is all that katpoint uses

The real pyephem computes observed place but katpoint always sets the
pressure to zero so we compute apparent places instead.
"""

from astropy.coordinates import get_moon
from astropy.coordinates import get_body
from astropy.coordinates import get_sun
from astropy.coordinates import EarthLocation
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import CIRS
from astropy.coordinates import ICRS
from astropy.coordinates import AltAz
from astropy.coordinates import SkyCoord
from astropy import coordinates
from astropy import units

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
        appt = icrs.transform_to(CIRS(obstime=obs.date._time))
        self.ra = astropy_angle(appt.ra, 'h')
        self.dec = astropy_angle(appt.dec)

        # ICRS to Az/El
        altaz = icrs.transform_to(AltAz(location=loc,
                obstime=obs.date._time, pressure=obs.pressure))
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
        return self.name + ',' + str(self._ra) + ',' + str(self._dec)

class Sun(Body):
    def __init__(self):
        Body.__init__(self)
        self.name = 'Sun'

    def compute(self, obs):
        loc = EarthLocation(lon=obs._lon.astropy_angle,
                lat=obs._lat.astropy_angle, height=obs.elevation)
        moon = get_sun(obs.date._time)
        icrs = moon.transform_to(ICRS)
        Body._compute(self, obs, icrs)


class Moon(Body):
    def __init__(self):
        Body.__init__(self)
        self.name = 'Moon'

    def compute(self, obs):
        loc = EarthLocation(lon=obs._lon.astropy_angle,
                lat=obs._lat.astropy_angle, height=obs.elevation)
        moon = get_moon(obs.date._time, loc)
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
            planet = get_body(self._name, obs.date._time, loc)
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
    def __init__(self, name):
        self.name = name
        Body.__init__(self)

    def compute(self, obs):
        self.a_ra = hours(0.0)
        self.a_dec = degrees(0.0)
        self.az = degrees(0.0)
        self.alt = degrees(0.0)

    def writedb(self):
        return self.name + 'E,2000,0.0,0.0,1.0,1,1,1,1,1,1'


def readtle(line1, line2, line3):
    return EarthSatellite(line1)
