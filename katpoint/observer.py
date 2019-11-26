################################################################################
# Copyright (c) 2009-2019, National Research Foundation (Square Kilometre Array)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

""" Replacement for ephem.Observer. """

import numpy as np

from astropy.time import Time
from astropy.coordinates import ICRS
from astropy.coordinates import AltAz
from astropy.coordinates import EarthLocation
from astropy import coordinates
from astropy import units

class Observer(object):
    """Represents a location.

    An observer object is just a date, a lon/lat and a height and has methods
    that calculates the sidereal time at the location and date/time and
    and for converting an az/el to an RA/Dec.

    Attributes
    ----------
    date : astropy.time.Time
        Date/time of observation

    lon : astropy.coordinates.Longitude
        Longitude of observer's location

    lat : astropy.coordinates.Latitude
        Latitude of observer's location

    elevation : float
        Elevation of observer in metres above mean sea level

    pressure : float
        Atmospheric pressure in mBar
    """
    def __init__(self):
        self.date = Time(Time.now(), scale='utc')
        self._lon = coordinates.Longitude(0.0, unit=units.deg)
        self._lat = coordinates.Latitude(0.0, unit=units.deg)
        self.elevation = 0.0
        self.pressure = 0.0

    @property
    def lon(self):
        return self._lon

    @lon.setter
    def lon(self, value):
        self._lon = value

    @property
    def long(self):
        return self._lon

    @long.setter
    def long(self, value):
        self._lon = value

    @property
    def lat(self):
        return self._lat

    @lat.setter
    def lat(self, value):
        self._lat = value

    def sidereal_time(self):
        """Returns the sidereal time as an astropy.Time.
        """
        time = Time(self.date, location=EarthLocation(lat=self._lat,
                lon=self._lon, height=self.elevation))
        return time.sidereal_time('apparent')

    def radec_of(self, az, alt):
        """Returns ICRS RA, Dec as astropy.coordinate.Angle.
        """
        loc = EarthLocation(lat=self._lat, lon=self._lon, height=self.elevation)
        altaz = AltAz(alt=alt, az=az, location=loc,
                obstime=self.date, pressure=self.pressure)
        radec = altaz.transform_to(ICRS)
        return radec.ra, radec.dec
