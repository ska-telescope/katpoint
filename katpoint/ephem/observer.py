""" Replacement for ephem.Observer
"""

from astropy.time import Time
from astropy.coordinates import CIRS
from astropy.coordinates import AltAz
from astropy.coordinates import EarthLocation

from .angle import degrees
from .angle import Angle
from .date import Date
from .date import now

class Observer(object):
    """Represents a location
    """
    def __init__(self):
        self.date = now()
        self.epoch = Date('2000/1/1 12:00:00')
        self.lon = degrees(0.0)
        self.lat = degrees(0.0)
        self.elevation = 0.0
        self.horizon = degrees(0.0)
        self.temp = 15.0
        self.pressure = 1010.0

    @property
    def lon(self):
        return self._lon

    @lon.setter
    def lon(self, value):
        self._lon = degrees(value)

    @property
    def long(self):
        return self._lon

    @long.setter
    def long(self, value):
        self._lon = degrees(value)

    @property
    def lat(self):
        return self._lat

    @lat.setter
    def lat(self, value):
        self._lat = degrees(value)

    def __str__(self):
        return ("<ephem.observer date='" + str(self.date) +
            " " + "epoch='" + str(self.epoch) + "' lon='" +
            str(self.lon) + "' lat='" + str(self.lat) + "' elevation=" +
            str(self.elevation) + "m horizon=" + str(self.horizon) +
            " temp=" + str(self.temp) + "C pressure=" + str(self.pressure) +
            "mBar>")

    def __repr__(self):
        return self.__str__()

    def sidereal_time(self):
        """Returns the sidereal time
        """
        loc = (self.lon._a, self.lat._a)
        t = Time(self.date._time, scale='utc', location=loc)
        st = t.sidereal_time('mean')
        return Angle(st)

    def radec_of(self, az, alt):
        """Returns topocentric apparent RA, Dec
        """
        loc = EarthLocation(self._lat._a, self._lon._a, self.elevation)
        altaz = AltAz(alt=az._a, az=alt._a, location=loc,
                obstime=self.date._time)
        radec = altaz.transform_to(CIRS)
        return Angle(radec.ra), Angle(radec.dec)
