""" Replacement for ephem.Angle
"""

import numpy
from astropy import coordinates
from astropy import units

class Angle():
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

def degrees(a):
    """ Create degrees (declination) Angle

    Parameters
    ----------
    a : An angle in hours as a float, an int or a sexagesimal string
    """
    if type(a) is float or type(a) is int:
        return Angle(str(numpy.rad2deg(a)) + 'd')
    else:
        return Angle(a + 'd')

def hours(a):
    """ Create an hours (RA) Angle

    Parameters
    ----------
    a : An angle in hours as a float, an int or a sexagesimal string
    """
    if type(a) is float or type(a) is int:
        return Angle(str(numpy.rad2deg(a/15.0)) + 'h')
    else:
        return Angle(a + 'h')
