""" Replacement for ephem.Date
"""

_djd = 15019.5

import time

from astropy.time import Time

class Date:
    """Represents a date and time

    This emulates the ephem.Date class but omits the "triple" functionality.
    """

    def __init__(self, d):
        if type(d) is float:
            self._time = Time(d + _djd, format='mjd', scale='utc')
        elif type(d) is tuple:
            s = '{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:f}'.format(d[0],d[1],
                    d[2],d[3],d[4],d[5])
            self._time = Time(s)
        else:
            self._time = Time(_decode(d))

    def __repr__(self):
        """Returns the date as a count of days since noon
        on the last day of 1899 (Dublin Julian Date).
        """
        return str(self._time.mjd - _djd)

    def __str__(self):
        """Returns the date as a string
        """
        return _encode(self._time.iso)

    def __eq__(self, d):
        return self._time == d._time

    def tuple(self):
        """ Returns the date as a tuple
        """
        dt = self._time.datetime
        return (dt.year, dt.month, dt.day, dt.hour, dt.minute,
                float(dt.second) + float(dt.microsecond)/1000000.0)

def _encode(iso):
    """Encodes an ISO date as a pyephem formatted date
    """
    yr = int(iso[0:4])
    mon = int(iso[5:7])
    day = int(iso[8:10])
    h = iso[11:13]
    m = iso[14:16]
    s = str(round(float(iso[17:23])))
    return (str(yr) + '/' + str(mon) + '/' + str(day) + ' ' +
            h + ':' + m + ':' + s)

def _decode(s):
    """Decodes a date string
    """
    # Look for a dot
    dot = s.find('.')
    if dot > 0:
        # fractional part of the seconds
        f = s[dot:]
        # date/time without fractional seconds
        s = s[:dot]
    else:
        f = '.0'

    # time without fractional seconds
    try:
        d = time.strptime(s, '%Y/%m/%d %H:%M:%S')
    except:
        try:
            d = time.strptime(s, '%Y/%m/%d %H:%M')
        except:
            try:
                d = time.strptime(s, '%Y/%m/%d %H')
            except:
                try:
                    d = time.strptime(s, '%Y/%m/%d')
                except:
                    try:
                        d = time.strptime(s, '%Y/%m')
                    except:
                        try:
                            d = time.strptime(s, '%Y')
                        except:
                            raise ValueError('unable to decode date string')

    # Convert to a unix time and add the fractional seconds
    u = time.mktime(d)

    # Back to a tuple
    d = time.localtime(u)

    return time.strftime('%Y-%m-%d %H:%M:%S',d) + f


def now():
    """ Create a Date representing 'now'
    """
    return Date(Time.now().mjd - _djd)
