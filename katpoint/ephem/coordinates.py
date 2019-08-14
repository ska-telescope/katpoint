"""Replacement for the ephem Galactic and Equatorial classes
"""

from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
from astropy.coordinates import ICRS
from astropy.coordinates import Galactic

from .angle import Angle

class Equatorial:
    def __init__(self, *args, epoch='2000'):
        if len(args) == 1:
            # Parameters are a body or an Equatorialor a Galactic.
            if type(args[0]) == Equatorial:
                self.ra = args[0].ra
                self.dec = args[0].dec
                self.epoch = epoch
            elif type(args[0]) == Galactic:
                pass
            else:
                self.ra = args[0].ra
                self.dec = args[0].dec
                self.epoch = epoch
        elif len(args) == 2:
            # Parameters are a pair of angles.
            self.ra = args[0]
            self.dec = args[1]
        else:
            raise runtimeError("wrong number of arguments")

    def get(self):
        return self.ra, self.dec


class Galactic:
    def __init__(self, *args):
        if len(args) == 1:
            # Parameters are a body.
            ra = args[0].ra
            dec = args[0].dec
            epoch = epoch[0].epoch
            fk5 = SkyCoord(ra=ra._a, dec=dec._a, epoch=epoch, frame='fk5')
            lonlat = fk5.transform_to(Galactic)
            lon = Angle(lonlat.l)
            lat = Angle(lonlat.b)
        elif len(args) == 2:
            # Parameters are a pair of angles.
            self.lon = args[0]
            self.lat = args[1]
        else:
            raise runtimeError("wrong number of arguments")

    def get(self):
        return self.lon, self.lat

    def to_radec(self):
        g = SkyCoord(l=self.lon._a, b=self.lat._a, frame='galactic')
        radec = g.transform_to(ICRS)
        return Angle(radec.ra), Angle(radec.dec)


