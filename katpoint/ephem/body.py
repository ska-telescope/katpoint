"""Replacement for the pyephem body classes.

We only implement ra, dec (CIRS), a_ra, a_dec (ICRS RA/dec) and alt, az
(topcentric) as that is all that katpoint uses

The real pyephem computes observed place but katpoint always sets the
pressure to zero so we compute apparent places instead.
"""

from astropy.coordinates import get_moon
from astropy.coordinates import get_body
from astropy.coordinates import EarthLocation
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import CIRS
from astropy.coordinates import ICRS
from astropy.coordinates import AltAz
from astropy.coordinates import SkyCoord
from astropy import coordinates
from astropy import units

from .constants import J2000
from .angle import degrees
from .angle import Angle

class Body(object):
    def __init__(self):
        self._epoch = J2000

    def compute(self, obs):
        self.a_ra  = self._ra
        self.a_dec = self._dec
        loc = EarthLocation(lon=obs._lon._a, lat=obs._lat._a,
                height=obs.elevation)

        # ICRS to CIRS
        icrs = SkyCoord(ra=self.a_ra._a, dec=self.a_dec._a, frame='icrs')
        appt = icrs.transform_to(CIRS(obstime=obs.date._time))
        rah = coordinates.Angle(appt.ra, unit=units.hourangle)
        self.ra = Angle(rah)
        self.dec = Angle(appt.dec)

        # ICRS to Az/El
        altaz = icrs.transform_to(AltAz(location=loc,
                obstime=obs.date._time, pressure=obs.pressure))
        self.az = Angle(altaz.az)
        self.alt = Angle(altaz.alt)


class FixedBody(Body):
    def __init__(self):
        Body.__init__(self)

    def compute(self, obs):
        Body.compute(self, obs)

class Sun(Body):
    def __init__(self):
        Body.__init__(self)
        self.name = 'Sun'

    def compute(self, obs):
        pass;

class Moon(Body):
    def __init__(self):
        Body.__init__(self)
        self.name = 'Moon'

    def compute(self, obs):
        loc = EarthLocation(obs._lat._a, obs._lon._a, obs.elevation)
        moon = get_moon(obs.date._time, loc)
        cirs = moon.transform_to(CIRS)
        self.a_ra = Angle(cirs.ra)
        self.a_dec = Angle(cirs.dec)
        icrs = moon.transform_to(ICRS)
        self.ra = Angle(icrs.ra)
        self.dec = Angle(icrs.dec)

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
        loc = EarthLocation(obs._lat._a, obs._lon._a, obs.elevation)
        with solar_system_ephemeris.set('builtin'):
            planet = get_body(self._name, obs.date._time, loc)
        cirs = planet.transform_to(CIRS)
        self.a_ra = Angle(cirs.ra)
        self.a_dec = Angle(cirs.dec)
        icrs = planet.transform_to(ICRS)
        self.ra = Angle(icrs.ra)
        self.dec = Angle(icrs.dec)

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
