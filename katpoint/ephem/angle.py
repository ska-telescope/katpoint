""" Replacement for ephem.Angle
"""

import numpy as np
from astropy import coordinates
from astropy import units

class Angle(object):
    """ Represents an angle
    """
    def __init__(self, a):
        self._a = coordinates.Angle(a)
        self.norm = self._a.wrap_at('360.0d').radian
        self.znorm = self._a.wrap_at('180.0d').radian

    def __repr__(self):
        return str(self._a.radian)

    def __str__(self):
        if self._a.unit == units.hourangle:
            return self._a.to_string(unit=units.hour, sep=':')
        else:
            return self._a.to_string(unit=units.degree, sep=':')

    def __float__(self):
        return self._a.radian

    def __add__(self, a):
        return self._a.radian + a

    def __radd__(self, a):
        return self._a.radian + a

    def __sub__(self, a):
        return self._a.radian - a._a.radian

    def __rsub__(self, a):
        return a - self._a.radian

    def __mul__(self, a):
        return self._a.radian * a

    def __neg__(self):
        return -self._a.radian

    def __truediv__(self, a):
        return self._a.radian / a

    def __lt__(self, a):
        if type(a) is Angle:
            return self._a.radian < a._a.radian
        else:
            return self._a.radian < a

    def __gt__(self, a):
        if type(a) is Angle:
            return self._a.radian > a._a.radian
        else:
            return self._a.radian > a

    def round(self):
        return np.round(self._a.radian)

    def rint(self):
        return np.rint(self._a.radian)

    def cos(a):
        return np.cos(a._a.radian)

    def sin(a):
        return np.sin(a._a.radian)

    def tan(a):
        return np.tan(a._a.radian)

    def __abs__(a):
        return np.abs(a._a.radian)

    def isfinite(a):
        return np.isfinite(a._a.radian)

def degrees(a):
    """ Create degrees (declination) Angle

    Parameters
    ----------
    a : An angle in radians or a sexagesimal string
    """
    if type(a) is str:
        return Angle(a + 'd')
    elif type(a) is Angle:
        return a
    else:
        return Angle(str(np.rad2deg(a)) + 'd')

def hours(a):
    """ Create an hours (RA) Angle

    Parameters
    ----------
    a : An angle in radians a sexagesimal string
    """
    if type(a) is str:
        return Angle(a + 'h')
    elif type(a) is Angle:
        return a
    else:
        return Angle(str(np.rad2deg(a/15.0)) + 'h')

def separation(p1, p2):
    """ Separation between two positions. (tuples of Angles)
    """
    c1 = coordinates.SkyCoord(p1[0]._a, p1[1]._a, frame='icrs')
    c2 = coordinates.SkyCoord(p2[0]._a, p2[1]._a, frame='icrs')
    s = c1.separation(c2)
    return Angle(s)
