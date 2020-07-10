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

import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord, ICRS, EarthLocation, Latitude, Longitude
from astropy.time import Time
import pytest

from katpoint.bodies import FixedBody, Sun, Moon, Mars, EarthSatellite, readtle


def _get_earth_satellite():
    name = ' GPS BIIA-21 (PRN 09) '
    line1 = '1 22700U 93042A   19266.32333151  .00000012  00000-0  10000-3 0  8057'
    line2 = '2 22700  55.4408  61.3790 0191986  78.1802 283.9935  2.00561720104282'
    return readtle(name, line1, line2)


class TestBody:
    """Test computing with various Bodies."""

    @pytest.mark.parametrize(
        "body, date_str, ra_str, dec_str, az_str, el_str",
        [
            (FixedBody(), '2020-01-01 00:00:00.000',
             '10:10:40.123', '40:20:50.567', '326:05:57.541', '51:21:20.0119'),
            #                                 326:05:54.8,     51:21:18.5  (PyEphem)
            (Mars(), '2020-01-01 00:00:00.000',
             '', '', '118:10:05.1129', '27:23:12.8499'),
            #         118:10:06.1,      27:23:13.3  (PyEphem)
            (Moon(), '2020-01-01 10:00:00.000',
             '', '', '127:15:17.1381', '60:05:10.2438'),
            #         127:15:23.6,      60:05:13.7  (PyEphem)
            (Sun(), '2020-01-01 10:00:00.000',
             '', '', '234:53:19.4835', '31:38:11.412'),
            #         234:53:20.8,      31:38:09.4  (PyEphem)
            (_get_earth_satellite(), '2019-09-23 07:45:36.000',
             '3:32:56.7813', '-2:04:35.4329', '280:32:29.675', '-54:06:50.7456'),
            # 3:32:59.21      -2:04:36.3       280:32:07.2      -54:06:14.4  (PyEphem)
        ]
    )
    def test_compute(self, body, date_str, ra_str, dec_str, az_str, el_str):
        """Test compute method"""
        lat = Latitude('10:00:00.000', unit=u.deg)
        lon = Longitude('80:00:00.000', unit=u.deg)
        date = Time(date_str)

        if isinstance(body, FixedBody):
            ra = Longitude(ra_str, unit=u.hour)
            dec = Latitude(dec_str, unit=u.deg)
            body._radec = SkyCoord(ra=ra, dec=dec, frame=ICRS)
        height = 4200.0 if isinstance(body, EarthSatellite) else 0.0
        body.compute(EarthLocation(lat=lat, lon=lon, height=height), date, 0.0)

        if ra_str and dec_str:
            assert body.a_radec.ra.to_string(sep=':', unit=u.hour) == ra_str
            assert body.a_radec.dec.to_string(sep=':') == dec_str
        assert body.altaz.az.to_string(sep=':') == az_str
        assert body.altaz.alt.to_string(sep=':') == el_str

    def test_earth_satellite(self):
        sat = _get_earth_satellite()
        # Check that the EarthSatellite object has the expect attribute values.
        assert str(sat._epoch) == '2019-09-23 07:45:35.842'
        assert sat._inc == np.deg2rad(55.4408)
        assert sat._raan == np.deg2rad(61.3790)
        assert sat._e == 0.0191986
        assert sat._ap == np.deg2rad(78.1802)
        assert sat._M == np.deg2rad(283.9935)
        assert sat._n == 2.0056172
        assert sat._decay == 1.2e-07
        assert sat._orbit == 10428
        assert sat._drag == 1.e-04

        # This is xephem database record that pyephem generates
        xephem = ' GPS BIIA-21 (PRN 09) ,E,9/23.32333151/2019| 6/15.3242/2019| 1/1.32422/2020,' \
                 '55.4408,61.379002,0.0191986,78.180199,283.9935,2.0056172,1.2e-07,10428,9.9999997e-05'

        rec = sat.writedb()
        assert rec.split(',')[0] == xephem.split(',')[0]
        assert rec.split(',')[1] == xephem.split(',')[1]

        assert (rec.split(',')[2].split('|')[0].split('/')[0]
                == xephem.split(',')[2].split('|')[0].split('/')[0])
        assert_allclose(float(rec.split(',')[2].split('|')[0].split('/')[1]),
                        float(xephem.split(',')[2].split('|')[0].split('/')[1]), rtol=0, atol=0.5e-7)
        assert (rec.split(',')[2].split('|')[0].split('/')[2]
                == xephem.split(',')[2].split('|')[0].split('/')[2])

        assert (rec.split(',')[2].split('|')[1].split('/')[0]
                == xephem.split(',')[2].split('|')[1].split('/')[0])
        assert_allclose(float(rec.split(',')[2].split('|')[1].split('/')[1]),
                        float(xephem.split(',')[2].split('|')[1].split('/')[1]), rtol=0, atol=0.5e-2)
        assert (rec.split(',')[2].split('|')[1].split('/')[2]
                == xephem.split(',')[2].split('|')[1].split('/')[2])

        assert (rec.split(',')[2].split('|')[2].split('/')[0]
                == xephem.split(',')[2].split('|')[2].split('/')[0])
        assert_allclose(float(rec.split(',')[2].split('|')[2].split('/')[1]),
                        float(xephem.split(',')[2].split('|')[2].split('/')[1]), rtol=0, atol=0.5e-2)
        assert (rec.split(',')[2].split('|')[2].split('/')[2]
                == xephem.split(',')[2].split('|')[2].split('/')[2])

        assert rec.split(',')[3] == xephem.split(',')[3]

        # pyephem adds spurious precision to these 3 fields
        assert rec.split(',')[4] == xephem.split(',')[4][:6]
        assert rec.split(',')[5][:7] == xephem.split(',')[5][:7]
        assert rec.split(',')[6] == xephem.split(',')[6][:5]

        assert rec.split(',')[7] == xephem.split(',')[7]
        assert rec.split(',')[8] == xephem.split(',')[8]
        assert rec.split(',')[9] == xephem.split(',')[9]
        assert rec.split(',')[10] == xephem.split(',')[10]
