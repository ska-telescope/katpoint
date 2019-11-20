"""Replacement for the ephem Galactic and Equatorial classes
"""

from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
from astropy.coordinates import ICRS
from astropy.coordinates import Galactic
from astropy import units
from astropy import coordinates
from astropy.time import Time

class Equatorial:
    def __init__(self, *args, epoch=Time(2000.0, format='jyear')):
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
                self.ra = args[0].a_ra
                self.dec = args[0].a_dec
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
            # Parameters are a body or an Equatorial.
            if type(args[0]) == Equatorial:
                ra = args[0].ra
                dec = args[0].dec
            else:
                ra = args[0].a_ra
                dec = args[0].a_dec
            fk5 = SkyCoord(ra=ra, dec=dec, frame='icrs')
            lonlat = fk5.transform_to('galactic')
            self.lon = lonlat.l
            self.lat = lonlat.b
        elif len(args) == 2:
            # Parameters are a pair of angles.
            self.lon = args[0]
            self.lat = args[1]
        else:
            raise runtimeError("wrong number of arguments")

    def get(self):
        return self.lon, self.lat

    def to_radec(self):
        g = SkyCoord(l=self.lon, b=self.lat, frame='galactic')
        radec = g.transform_to(ICRS)
        return radec.ra, radec.dec


