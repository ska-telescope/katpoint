################################################################################
# Copyright (c) 2019-2023, National Research Foundation (SARAO)
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

import astropy.units as u
import pytest
from astropy import __version__ as astropy_version
from astropy.coordinates import (
    ICRS,
    AltAz,
    Angle,
    EarthLocation,
    Galactic,
    Latitude,
    Longitude,
    SkyCoord,
)
from astropy.time import Time
from packaging.version import Version

from katpoint.body import (
    Body,
    EarthSatelliteBody,
    FixedBody,
    GalacticBody,
    SolarSystemBody,
    StationaryBody,
)

try:
    from skyfield.api import EarthSatellite, Topos, load
except ImportError:
    HAS_SKYFIELD = False
else:
    HAS_SKYFIELD = True

from .helper import check_separation


def _get_fixed_body(ra_str, dec_str, distance=None):
    ra = Angle(ra_str, unit=u.hour)
    dec = Angle(dec_str, unit=u.deg)
    return FixedBody(SkyCoord(ra=ra, dec=dec, distance=distance))


TLE_NAME = "GPS BIIA-21 (PRN 09)"
TLE_LINE1 = "1 22700U 93042A   19266.32333151  .00000012  00000-0  10000-3 0  8057"
TLE_LINE2 = "2 22700  55.4408  61.3790 0191986  78.1802 283.9935  2.00561720104282"
TLE_TS = "2019-09-23 07:45:36.000"
TLE_AZ = "280:32:29.6594d"  # Astropy 5.3
# 1.      280:32:29.6594   Astropy 4.3 (0.0003" error)
# 2.      280:32:28.6175   Skyfield 1.46 (3.7" error)
# 3.      280:32:28.1892   Astropy 4.1 (4.1" error)
# 4.      280:32:29.675    Astropy 4.0.1 + PyOrbital for TEME (5.3" error)
# 5.      280:32:07.2      PyEphem 3.7.7.0 (33.7" error)
TLE_EL = "-54:06:29.1901d"  # Astropy 5.3
# 1.      -54:06:29.1898   Astropy 4.3
# 2.      -54:06:32.8635   Skyfield 1.46
# 3.      -54:06:33.1950   Astropy 4.1
# 4.      -54:06:34.5473   Astropy 4.0.1 + PyOrbital for TEME
# 5.      -54:05:58.2      PyEphem 3.7.7.0
LOCATION = EarthLocation(lat=10.0, lon=80.0, height=0.0)
astropy_version = Version(astropy_version)


# The reference coordinate values below are based on Astropy 5.3.
# It ships with a more accurate IERS_B table based on ITRF2020 and differs
# by about 1 mas from the previous Astropy versions starting from 4.3.
# Astropy 5.0 introduced a more accurate Moon model.
# Astropy 4.3 introduced topocentric CIRS in PR astropy/astropy#10994
# which improved (az, el) for nearby objects, so coordinates in older
# Astropy versions differ by more.
# The test tolerances are adjusted so that the tests still pass on Astropy 4.1.
# The accuracy is capped to 0.00005" since we specify azel arcsec to 4 decimal places.
@pytest.mark.parametrize(
    "body, date_str, ra_str, dec_str, az_str, el_str, tol, ref_astropy_ver",
    [
        (
            _get_fixed_body("10:10:40.123", "40:20:50.567"),
            "2020-01-01 00:00:00.000",
            "10:10:40.123h",
            "40:20:50.567d",
            "326:05:57.5414d",
            "51:21:20.0122d",
            1 * u.mas,
            "5.3",
        ),
        # 10:10:40.12h     40:20:50.6d      326:05:54.8d       51:21:18.5d  (PyEphem)
        # Adjust time by UT1-UTC=-0.177:    326:05:57.1d       51:21:19.9d  (PyEphem)
        (
            _get_fixed_body("10:10:40.123", "40:20:50.567", 0 * u.m),
            "2020-01-01 00:00:00.000",
            "18:43:01.13546h",
            "-23:04:13.1204d",
            "111:27:59.7729d",
            "-13:52:32.0920d",
            60 * u.mas,
            "5.3",
        ),
        # A distance of 0 m takes us to the barycentre,
        # so way different (ra, dec); cf. Sun below
        (
            SolarSystemBody("Mars"),
            "2020-01-01 00:00:00.000",
            "15:43:47.34130h",
            "-19:23:08.1338d",
            "118:10:05.1118d",
            "27:23:12.8449d",
            1 * u.mas,
            "5.3",
        ),
        # 15:43:47.22h       -19:23:07.0d       118:10:06.1d       27:23:13.3d
        # (PyEphem)
        (
            SolarSystemBody("Moon"),
            "2020-01-01 10:00:00.000",
            "23:35:44.14405h",
            "-8:32:55.3779d",
            "127:15:16.5836d",
            "60:05:10.4217d",
            320 * u.mas,
            "5.3",
        ),
        # 23:34:17.02h       -8:16:33.4d       127:15:23.6d       60:05:13.7d  (PyEphem)
        # The Moon radec differs by quite a bit (16') because PyEphem's
        # astrometric radec is geocentric while katpoint's version is topocentric
        # (FWIW, Skyfield has both). Katpoint's geocentric astrometric radec is
        # 23:34:17.5082, -8:16:28.6389.
        (
            SolarSystemBody("Sun"),
            "2020-01-01 10:00:00.000",
            "18:44:13.36204h",
            "-23:02:54.8156d",
            "234:53:19.4757d",
            "31:38:11.4257d",
            160 * u.mas,
            "5.3",
        ),
        # 18:44:13.84h       -23:02:51.2d       234:53:20.8d       31:38:09.4d
        # (PyEphem)
        (
            EarthSatelliteBody.from_tle(TLE_LINE1, TLE_LINE2),
            TLE_TS,
            "3:32:58.17409h",
            "-2:04:30.0658d",
            TLE_AZ,
            TLE_EL,
            4200 * u.mas,
            "5.3",
        ),
        # 3:33:00.26h       -2:04:32.2d  (PyEphem)
        (
            StationaryBody("127:15:17.1418", "60:05:10.5475"),
            "2020-01-01 10:00:00.000",
            "23:35:44.12588h",
            "-8:32:55.5217d",
            "127:15:17.1418d",
            "60:05:10.5475d",
            1 * u.mas,
            "5.3",
        ),
        # 23:35:44.31h       -8:32:55.3d       127:15:17.1d       60:05:10.5d  (PyEphem)
    ],
)
def test_compute(body, date_str, ra_str, dec_str, az_str, el_str, tol, ref_astropy_ver):
    """Test `body.compute()` for the two ends of the coordinate chain."""
    obstime = Time(date_str)
    is_moon = body.default_name == "Moon"
    accurate = astropy_version >= Version(ref_astropy_ver)
    if accurate:
        # Accuracy is roughly capped to 0.0001" since we specify arcseconds
        # to 4 decimal places and hourangle seconds to 5 decimal places, and
        # separation involves the quadrature sum of two coordinate offsets.
        tol = 0.1 * u.mas
    # Go to the bottom of the coordinate chain: (az, el)
    altaz = body.compute(AltAz(obstime=obstime, location=LOCATION), obstime, LOCATION)
    check_separation(altaz, az_str, el_str, tol)
    # Go to the top of the coordinate chain: astrometric (ra, dec)
    radec = body.compute(ICRS(), obstime, LOCATION, to_celestial_sphere=True)
    check_separation(
        radec, ra_str, dec_str, 1 * u.mas if not accurate and not is_moon else tol
    )
    # Check that astrometric (ra, dec) results in the same (az, el) as a double-check
    altaz2 = radec.transform_to(AltAz(obstime=obstime, location=LOCATION))
    check_separation(altaz2, az_str, el_str, tol)


def test_fixed():
    """Test basic properties of fixed body."""
    body = _get_fixed_body("10:10:40.123", "40:20:50.567")
    assert body.tag == "radec"
    assert body.default_name
    assert body.coord.ra == Longitude("10:10:40.123h")
    assert body.coord.dec == Latitude("40:20:50.567d")


def test_galactic():
    """Test basic properties of galactic body."""
    body = GalacticBody(Galactic(l="-10d", b="20d"))
    assert body.tag == "gal"
    assert body.default_name
    assert body.coord.l == Longitude("-10d")  # noqa: E741
    assert body.coord.b == Latitude("20d")


def test_solar_system():
    """Test basic properties of solar system body."""
    body = SolarSystemBody("Venus")
    assert body.tag == "special"
    assert body.default_name


@pytest.mark.skipif(not HAS_SKYFIELD, reason="Skyfield is not installed")
def test_earth_satellite_vs_skyfield():
    """Test whether Skyfield agrees with katpoint on satellite (az, el)."""
    ts = load.timescale()
    satellite = EarthSatellite(TLE_LINE1, TLE_LINE2, TLE_NAME, ts)
    antenna = Topos(
        latitude_degrees=LOCATION.lat.deg,
        longitude_degrees=LOCATION.lon.deg,
        elevation_m=LOCATION.height.value,
    )
    obstime = Time(TLE_TS)
    t = ts.from_astropy(obstime)
    towards_sat = (satellite - antenna).at(t)
    alt, az, _ = towards_sat.altaz()
    altaz = AltAz(
        alt=alt.to(u.rad), az=az.to(u.rad), obstime=obstime, location=LOCATION
    )
    check_separation(altaz, TLE_AZ, TLE_EL, 4.0 * u.arcsec)


def _check_edb_E(sat, epoch_iso, inc, raan, e, ap, M, n, decay, drag):
    """Check SGP4 object and EDB versions of standard orbital parameters."""
    epoch = Time(sat.jdsatepoch, sat.jdsatepochF, format="jd")
    assert epoch.iso == epoch_iso
    assert sat.inclo * u.rad == inc * u.deg
    assert sat.nodeo * u.rad == raan * u.deg
    assert sat.ecco == e
    assert sat.argpo * u.rad == ap * u.deg
    assert sat.mo * u.rad == M * u.deg
    sat_n = (sat.no_kozai * u.rad / u.minute).to(u.cycle / u.day)
    assert sat_n.value == pytest.approx(n, abs=1e-15)
    assert sat.ndot * u.rad / u.minute**2 == decay * u.cycle / u.day**2
    assert sat.bstar == drag


def test_earth_satellite():
    """Test basic properties of satellite body and conversion to/from EDB."""
    body = EarthSatelliteBody.from_tle(TLE_LINE1, TLE_LINE2)
    assert body.tag == "tle"
    assert body.default_name
    assert body.to_tle() == (TLE_LINE1, TLE_LINE2)
    # Check that the EarthSatelliteBody object has the expected attribute values
    _check_edb_E(
        body.satellite,
        epoch_iso="2019-09-23 07:45:35.842",
        inc=55.4408,
        raan=61.3790,
        e=0.0191986,
        ap=78.1802,
        M=283.9935,
        n=2.0056172,
        decay=1.2e-07,
        drag=1.0e-04,
    )
    assert body.satellite.revnum == 10428
    # This is the XEphem database record that PyEphem generates (without name)
    xephem = (
        ",E,9/23.32333151/2019| 6/15.3242/2019| 1/1.32422/2020,"
        "55.4408,61.379002,0.0191986,78.180199,283.9935,"
        "2.0056172,1.2e-07,10428,9.9999997e-05"
    )
    assert body.to_edb() == xephem
    # Check some round-tripping
    body2 = Body.from_edb(xephem)
    assert body2.tag == "xephem tle"
    assert body2.default_name
    assert isinstance(body2, EarthSatelliteBody)
    assert body2.to_edb() == xephem


def test_stationary():
    """Test basic properties of stationary body."""
    body = StationaryBody("20d", "30d")
    assert body.tag == "azel"
    assert body.default_name
    assert body.coord.az == Longitude("20d")
    assert body.coord.alt == Latitude("30d")


def test_fixed_edb():
    """Test that we can construct fixed body from EDB."""
    record = "Sadr,f|S|F8,20:22:13.7|2.43,40:15:24|-0.93,2.23,2000,0"
    body = Body.from_edb(record)
    assert isinstance(body, FixedBody)
    assert body.tag == "radec"
    assert body.default_name
    assert body.coord.ra == Longitude("20:22:13.7h")
    assert body.coord.dec == Latitude("40:15:24d")
    assert body.to_edb() == ",f,20:22:13.7,40:15:24"  # no name or proper motion
