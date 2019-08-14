"""Replacement for the ephem Galactic and Equatorial classes
"""

from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
from astropy.coordinates import ICRS
from astropy.coordinates import Galactic
from astropy import units
from astropy import coordinates

from .angle import Angle
from .angle import degrees
from .constants import J2000

class Equatorial:
    def __init__(self, *args, epoch=J2000):
        if len(args) == 1:
            # Parameters are a body or an Equatorial or a Galactic.
            if type(args[0]) == Equatorial:
                self.ra = args[0].ra
                self.dec = args[0].dec
            elif type(args[0]) == Galactic:
                radec = args[0].to_radec()
                self.ra = radec[0]
                self.dec = radec[1]
            else:
                self.ra = args[0].ra
                self.dec = args[0].dec
        elif len(args) == 2:
            # Parameters are a pair of angles.
            self.ra = args[0]
            self.dec = args[1]
        else:
            raise runtimeError("wrong number of arguments")
        self._epoch = epoch

    def get(self):
        return self.ra, self.dec


class Galactic:
    def __init__(self, *args):
        if len(args) == 1:
            # Parameters are a body.
            ra = args[0].ra
            dec = args[0].dec
            epoch = args[0]._epoch
            fk5 = SkyCoord(ra=ra._a, dec=dec._a, frame='fk5')
            lonlat = fk5.transform_to('galactic')
            self.lon = Angle(lonlat.l)
            self.lat = Angle(lonlat.b)
        elif len(args) == 2:
            # Parameters are a pair of floats.
            self.lon = degrees(args[0])
            self.lat = degrees(args[1])
        else:
            raise runtimeError("wrong number of arguments")

    def get(self):
        return self.lon, self.lat

    def to_radec(self):
        g = SkyCoord(l=self.lon._a, b=self.lat._a, frame='galactic')
        radec = g.transform_to(ICRS)
        rah = coordinates.Angle(radec.ra, unit=units.hourangle)
        return Angle(rah), Angle(radec.dec)


