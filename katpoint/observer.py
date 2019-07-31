""" Replacement for ephem.Observer
"""

from astropy.time import Time

from angle import degrees
from angle import Angle
from date import Date
from date import now

class Observer:
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