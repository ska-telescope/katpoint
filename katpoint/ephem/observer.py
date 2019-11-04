""" Replacement for ephem.Observer
"""

from astropy.time import Time
from astropy.coordinates import CIRS
from astropy.coordinates import AltAz
from astropy.coordinates import EarthLocation
from astropy import coordinates
from astropy import units

from .angle import degrees
from .angle import astropy_angle
from .constants import J2000

class Observer(object):
    """Represents a location
    """
    def __init__(self):
        self.date = Time(Time.now(), scale='utc')
        self.epoch = J2000
        self.lon = degrees(0.0)
        self.lat = degrees(0.0)
        self.elevation = 0.0
        self.horizon = degrees(0.0)
        self.temp = 15.0
        self.pressure = 0.0

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
        loc = EarthLocation(lat=self._lat.astropy_angle,
                lon=self._lon.astropy_angle, height=self.elevation)
        t = Time(self.date, location=loc)
        st = t.sidereal_time('apparent')
        return astropy_angle(st, 'h')

    def radec_of(self, az, alt):
        """Returns topocentric apparent RA, Dec
        """
        loc = EarthLocation(lat=self._lat.astropy_angle,
                lon=self._lon.astropy_angle, height=self.elevation)
        altaz = AltAz(alt=alt.astropy_angle, az=az.astropy_angle, location=loc,
                obstime=self.date, pressure=self.pressure)
        radec = altaz.transform_to(CIRS)
        return astropy_angle(radec.ra, 'h'), astropy_angle(radec.dec)
