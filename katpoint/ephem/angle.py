from astropy import units
from astropy import coordinates

class Angle(float):

    def __new__(cls, value, kind):
        if kind not in ('d', 'h'):
            raise ValueError("kind must be 'd' or 'h'")
        instance = super().__new__(cls, value)
        instance.kind = kind
        return instance

    @property
    def norm(self):
        ang = coordinates.Angle(float(self), unit=units.rad)
        val = ang.wrap_at('360.0d').radian
        return Angle(val, self.kind)

    @property
    def znorm(self):
        ang = coordinates.Angle(float(self), unit=units.rad)
        val = ang.wrap_at('180.0d').radian
        return Angle(val, self.kind)

    def __str__(self):
        ang = coordinates.Angle(float(self), unit=units.rad)
        if self.kind == 'd':
            out = ang.to_string(unit=units.deg, sep=':', precision=1)
        elif self.kind == 'h':
            out = ang.to_string(unit=units.hourangle, sep=':', precision=2)
        return out

    def __pos__(self):
        return Angle(+float(self), self.kind)

    def __neg__(self):
        return Angle(-float(self), self.kind)

def degrees(a):
    if isinstance(a, str):
        val = coordinates.Angle(a, unit=units.deg).radian
    elif isinstance(a, Angle):
        val = float(a)
    else:
        val = a
    return Angle(val, 'd')

def hours(a):
    if isinstance(a, str):
        val = coordinates.Angle(a, unit=units.hourangle).radian
    elif isinstance(a, Angle):
        val = float(a)
    else:
        val = a
    return Angle(val, 'h')

def separation(p1, p2):
    """ Separation between two positions (tuples of Angles).
    """
    c1 = coordinates.SkyCoord(p1[0], p1[1], unit=units.rad, frame='icrs')
    c2 = coordinates.SkyCoord(p2[0], p2[1], unit=units.rad, frame='icrs')
    s = c1.separation(c2)
    return Angle(s.radian, 'd')
