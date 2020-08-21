################################################################################
# Copyright (c) 2009-2020, National Research Foundation (SARAO)
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

"""Tests for the body module.

The values in the comments are the results from the same tests run on the
pyephem package.

"""

import pytest
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, ICRS, AltAz
from astropy.coordinates import EarthLocation, Latitude, Longitude

from katpoint.body import FixedBody, SolarSystemBody, EarthSatelliteBody
from katpoint.test.helper import check_separation

try:
    from skyfield.api import load, EarthSatellite, Topos
except ImportError:
    HAS_SKYFIELD = False
else:
    HAS_SKYFIELD = True


def _get_fixed_body(ra_str, dec_str):
    ra = Longitude(ra_str, unit=u.hour)
    dec = Latitude(dec_str, unit=u.deg)
    return FixedBody('name', SkyCoord(ra=ra, dec=dec, frame=ICRS))


TLE_NAME = 'GPS BIIA-21 (PRN 09)'
TLE_LINE1 = '1 22700U 93042A   19266.32333151  .00000012  00000-0  10000-3 0  8057'
TLE_LINE2 = '2 22700  55.4408  61.3790 0191986  78.1802 283.9935  2.00561720104282'
TLE_TS = '2019-09-23 07:45:36.000'
TLE_AZ = '280:32:28.1892d'
# 1.      280:32:28.6053   Skyfield (0.43" error, was 0.23" with WGS84)
# 2.      280:32:29.675    Astropy 4.0.1 + PyOrbital for TEME (1.82" error)
# 3.      280:32:07.2      PyEphem (37" error)
TLE_EL = '-54:06:49.3936d'
# 1.      -54:06:49.0358   Skyfield
# 2.      -54:06:50.7456   Astropy 4.0.1 + PyOrbital for TEME
# 3.      -54:06:14.4      PyEphem
TLE_LOCATION = EarthLocation(lat=10.0, lon=80.0, height=4200.0)
LOCATION = EarthLocation(lat=10.0, lon=80.0, height=0.0)


@pytest.mark.parametrize(
    "body, date_str, ra_str, dec_str, az_str, el_str, tol",
    [
        (_get_fixed_body('10:10:40.123', '40:20:50.567'), '2020-01-01 00:00:00.000',
         '10:10:40.123h', '40:20:50.567d', '326:05:57.541d', '51:21:20.0119d', 1 * u.mas),
        # 10:10:40.12h     40:20:50.6d      326:05:54.8d      51:21:18.5d  (PyEphem)
        # Adjust time by UT1-UTC=-0.177:    326:05:57.1d      51:21:19.9  (PyEphem)
        (SolarSystemBody('Mars'), '2020-01-01 00:00:00.000',
         '14:05:58.9201h', '-12:13:51.9009d', '118:10:05.1129d', '27:23:12.8499d', 1 * u.mas),
        # (PyEphem radec is geocentric)        118:10:06.1d       27:23:13.3d  (PyEphem)
        (SolarSystemBody('Moon'), '2020-01-01 10:00:00.000',
         '6:44:11.9332h', '23:02:08.402d', '127:15:17.1418d', '60:05:10.5475d', 1 * u.mas),
        # (PyEphem radec is geocentric)     127:15:23.6d       60:05:13.7d  (PyEphem)
        (SolarSystemBody('Sun'), '2020-01-01 10:00:00.000',
         '7:56:36.7964h', '20:53:59.4553d', '234:53:19.4762d', '31:38:11.4248d', 1 * u.mas),
        # (PyEphem radec is geocentric)      234:53:20.8d       31:38:09.4d  (PyEphem)
        (EarthSatelliteBody.from_tle(TLE_NAME, TLE_LINE1, TLE_LINE2), TLE_TS,
         '0:00:38.5009h', '00:03:56.0093d', TLE_AZ, TLE_EL, 1 * u.mas),
    ]
)
def test_compute(body, date_str, ra_str, dec_str, az_str, el_str, tol):
    """Test compute method"""
    obstime = Time(date_str)
    location = TLE_LOCATION if isinstance(body, EarthSatelliteBody) else LOCATION
    radec = body.compute(ICRS(), obstime, location)
    check_separation(radec, ra_str, dec_str, tol)
    altaz = body.compute(AltAz(obstime=obstime, location=location), obstime, location)
    check_separation(altaz, az_str, el_str, tol)


@pytest.mark.skipif(not HAS_SKYFIELD, reason="Skyfield is not installed")
def test_earth_satellite_vs_skyfield():
    ts = load.timescale()
    satellite = EarthSatellite(TLE_LINE1, TLE_LINE2, TLE_NAME, ts)
    antenna = Topos(latitude_degrees=TLE_LOCATION.lat.deg,
                    longitude_degrees=TLE_LOCATION.lon.deg,
                    elevation_m=TLE_LOCATION.height.value)
    obstime = Time(TLE_TS)
    t = ts.from_astropy(obstime)
    towards_sat = (satellite - antenna).at(t)
    alt, az, distance = towards_sat.altaz()
    altaz = AltAz(alt=Latitude(alt.radians, unit=u.rad),
                  az=Longitude(az.radians, unit=u.rad),
                  obstime=obstime, location=TLE_LOCATION)
    check_separation(altaz, TLE_AZ, TLE_EL, 0.5 * u.arcsec)


def test_earth_satellite():
    body = EarthSatelliteBody.from_tle(TLE_NAME, TLE_LINE1, TLE_LINE2)
    sat = body.satellite
    # Check that the EarthSatelliteBody object has the expected attribute values
    epoch = Time(sat.jdsatepoch, sat.jdsatepochF, format='jd')
    assert epoch.iso == '2019-09-23 07:45:35.842'
    assert sat.inclo * u.rad == 55.4408 * u.deg
    assert sat.nodeo * u.rad == 61.3790 * u.deg
    assert sat.ecco == 0.0191986
    assert sat.argpo * u.rad == 78.1802 * u.deg
    assert sat.mo * u.rad == 283.9935 * u.deg
    assert 2.0056172 * u.cycle / u.day == sat.no_kozai * u.rad / u.minute
    assert sat.ndot * u.rad / u.minute ** 2 == 1.2e-07 * u.cycle / u.day ** 2
    assert sat.revnum == 10428
    assert sat.bstar == 1.e-04
    # This is the XEphem database record that PyEphem generates
    xephem = ('GPS BIIA-21 (PRN 09),E,'
              '9/23.32333151/2019| 6/15.3242/2019| 1/1.32422/2020,'
              '55.4408,61.379002,0.0191986,78.180199,283.9935,'
              '2.0056172,1.2e-07,10428,9.9999997e-05')
    assert body.writedb() == xephem
