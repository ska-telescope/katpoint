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
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, ICRS, AltAz, EarthLocation, Angle

from katpoint.body import Body, FixedBody, SolarSystemBody, EarthSatelliteBody, to_angle
from katpoint.test.helper import check_separation

try:
    from skyfield.api import load, EarthSatellite, Topos
except ImportError:
    HAS_SKYFIELD = False
else:
    HAS_SKYFIELD = True


@pytest.mark.parametrize("angle, angle_deg", [('10:00:00', 10), ('10:45:00', 10.75), ('10.0', 10),
                                              ((10 * u.deg).to_value(u.rad), pytest.approx(10)),
                                              ('10d00m00s', 10), ((10, 0, 0), 10),
                                              ('10h00m00s', pytest.approx(150))])
def test_angle_from_degrees(angle, angle_deg):
    assert to_angle(angle, sexagesimal_unit=u.deg).deg == angle_deg


@pytest.mark.parametrize("angle, angle_hour", [('10:00:00', 10), ('10:45:00', 10.75),
                                               ('150.0', pytest.approx(10)),
                                               ((150 * u.deg).to_value(u.rad), pytest.approx(10)),
                                               ('10h00m00s', 10), ((10, 0, 0), 10),
                                               ('10d00m00s', pytest.approx(10 / 15))])
def test_angle_from_hours(angle, angle_hour):
    assert to_angle(angle, sexagesimal_unit=u.hour).hour == angle_hour


def _get_fixed_body(ra_str, dec_str):
    return FixedBody('name', SkyCoord(ra=Angle(ra_str, unit=u.hour),
                                      dec=Angle(dec_str, unit=u.deg)))


TLE_NAME = 'GPS BIIA-21 (PRN 09)'
TLE_LINE1 = '1 22700U 93042A   19266.32333151  .00000012  00000-0  10000-3 0  8057'
TLE_LINE2 = '2 22700  55.4408  61.3790 0191986  78.1802 283.9935  2.00561720104282'
TLE_TS = '2019-09-23 07:45:36.000'
TLE_AZ = '280:32:28.1892d'
# 1.      280:32:28.6053   Skyfield (0.43" error, was 0.23" with WGS84)
# 2.      280:32:29.675    Astropy 4.0.1 + PyOrbital for TEME (1.61" error)
# 3.      280:32:07.2      PyEphem (37" error)
TLE_EL = '-54:06:33.1950d'
# 1.      -54:06:32.8374   Skyfield
# 2.      -54:06:34.5473   Astropy 4.0.1 + PyOrbital for TEME
# 3.      -54:05:58.2      PyEphem
LOCATION = EarthLocation(lat=10.0, lon=80.0, height=0.0)


@pytest.mark.parametrize(
    "body, date_str, ra_str, dec_str, az_str, el_str, tol",
    [
        (_get_fixed_body('10:10:40.123', '40:20:50.567'), '2020-01-01 00:00:00.000',
         '10:10:40.123h', '40:20:50.567d', '326:05:57.541d', '51:21:20.0119d', 1 * u.mas),
        # 10:10:40.12h     40:20:50.6d      326:05:54.8d      51:21:18.5d  (PyEphem)
        # Adjust time by UT1-UTC=-0.177:    326:05:57.1d      51:21:19.9  (PyEphem)
        (SolarSystemBody('Mars'), '2020-01-01 00:00:00.000',
         '14:05:58.9201h', '-12:13:51.9009d', '118:10:05.1121d', '27:23:12.8454d', 1 * u.mas),
        # (PyEphem radec is geocentric)        118:10:06.1d       27:23:13.3d  (PyEphem)
        (SolarSystemBody('Moon'), '2020-01-01 10:00:00.000',
         '6:44:11.9332h', '23:02:08.4027d', '127:15:17.1418d', '60:05:10.5475d', 1 * u.mas),
        # (PyEphem radec is geocentric)     127:15:23.6d       60:05:13.7d  (PyEphem)
        (SolarSystemBody('Sun'), '2020-01-01 10:00:00.000',
         '7:56:36.7961h', '20:53:59.4561d', '234:53:19.4763d', '31:38:11.4248d', 1 * u.mas),
        # (PyEphem radec is geocentric)      234:53:20.8d       31:38:09.4d  (PyEphem)
        (EarthSatelliteBody.from_tle(TLE_NAME, TLE_LINE1, TLE_LINE2), TLE_TS,
         '0:00:38.5009h', '00:03:56.0093d', TLE_AZ, TLE_EL, 1 * u.mas),
    ]
)
def test_compute(body, date_str, ra_str, dec_str, az_str, el_str, tol):
    """Test compute method"""
    obstime = Time(date_str)
    radec = body.compute(ICRS(), obstime, LOCATION)
    check_separation(radec, ra_str, dec_str, tol)
    altaz = body.compute(AltAz(obstime=obstime, location=LOCATION), obstime, LOCATION)
    check_separation(altaz, az_str, el_str, tol)


@pytest.mark.skipif(not HAS_SKYFIELD, reason="Skyfield is not installed")
def test_earth_satellite_vs_skyfield():
    ts = load.timescale()
    satellite = EarthSatellite(TLE_LINE1, TLE_LINE2, TLE_NAME, ts)
    antenna = Topos(latitude_degrees=LOCATION.lat.deg,
                    longitude_degrees=LOCATION.lon.deg,
                    elevation_m=LOCATION.height.value)
    obstime = Time(TLE_TS)
    t = ts.from_astropy(obstime)
    towards_sat = (satellite - antenna).at(t)
    alt, az, distance = towards_sat.altaz()
    altaz = AltAz(alt=alt.to(u.rad), az=az.to(u.rad), obstime=obstime, location=LOCATION)
    check_separation(altaz, TLE_AZ, TLE_EL, 0.5 * u.arcsec)


def _check_edb_E(sat, epoch_iso, inc, raan, e, ap, M, n, decay, drag):
    """Check SGP4 object and EDB versions of standard orbital parameters."""
    epoch = Time(sat.jdsatepoch, sat.jdsatepochF, format='jd')
    assert epoch.iso == epoch_iso
    assert sat.inclo * u.rad == inc * u.deg
    assert sat.nodeo * u.rad == raan * u.deg
    assert sat.ecco == e
    assert sat.argpo * u.rad == ap * u.deg
    assert sat.mo * u.rad == M * u.deg
    sat_n = (sat.no_kozai * u.rad / u.minute).to(u.cycle / u.day)
    assert sat_n.value == pytest.approx(n, abs=1e-15)
    assert sat.ndot * u.rad / u.minute ** 2 == decay * u.cycle / u.day ** 2
    assert sat.bstar == drag


def test_earth_satellite():
    body = EarthSatelliteBody.from_tle(TLE_NAME, TLE_LINE1, TLE_LINE2)
    assert body.to_tle() == (TLE_LINE1, TLE_LINE2)
    # Check that the EarthSatelliteBody object has the expected attribute values
    _check_edb_E(body.satellite, epoch_iso='2019-09-23 07:45:35.842',
                 inc=55.4408, raan=61.3790, e=0.0191986, ap=78.1802, M=283.9935,
                 n=2.0056172, decay=1.2e-07, drag=1.e-04)
    assert body.satellite.revnum == 10428
    # This is the XEphem database record that PyEphem generates
    xephem = ('GPS BIIA-21 (PRN 09),E,'
              '9/23.32333151/2019| 6/15.3242/2019| 1/1.32422/2020,'
              '55.4408,61.379002,0.0191986,78.180199,283.9935,'
              '2.0056172,1.2e-07,10428,9.9999997e-05')
    assert body.to_edb() == xephem
    # Check some round-tripping
    body2 = Body.from_edb(xephem)
    assert isinstance(body2, EarthSatelliteBody)
    assert body2.to_edb() == xephem


def test_star():
    record = 'Sadr,f|S|F8,20:22:13.7|2.43,40:15:24|-0.93,2.23,2000,0'
    e = Body.from_edb(record)
    assert isinstance(e, FixedBody)
    assert e.name == 'Sadr'
    assert e.coord.ra.to_string(sep=':', unit='hour') == '20:22:13.7'
    assert e.coord.dec.to_string(sep=':', unit='deg') == '40:15:24'
