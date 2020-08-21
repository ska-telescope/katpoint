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

"""Tests for the stars module."""

import astropy.units as u
from astropy.time import Time

from katpoint.stars import readdb
from katpoint.body import EarthSatelliteBody, FixedBody


def test_earth_satellite():
    record = 'GPS BIIA-21 (PR,E,9/23.32333151/2019| 6/15.3242/2019| 1/1.32422/2020,' \
        '55.4408,61.379002,0.0191986,78.180199,283.9935,2.0056172,1.2e-07,10428,9.9999997e-05'

    e = readdb(record)
    assert isinstance(e, EarthSatelliteBody)
    assert e.name == 'GPS BIIA-21 (PR'
    sat = e.satellite
    epoch = Time(sat.jdsatepoch, sat.jdsatepochF, format='jd')
    assert epoch.iso == '2019-09-23 07:45:35.842'
    assert sat.inclo * u.rad == 55.4408 * u.deg
    assert sat.nodeo * u.rad == 61.379002 * u.deg
    assert sat.ecco == 0.0191986
    assert sat.argpo * u.rad == 78.180199 * u.deg
    assert sat.mo * u.rad == 283.9935 * u.deg
    assert sat.no_kozai * u.rad / u.minute == 2.0056172 * u.cycle / u.day
    assert sat.ndot * u.rad / u.minute ** 2 == 1.2e-07 * u.cycle / u.day ** 2
    assert e.orbit_number == 10428  # XXX This should eventually be sat.revnum
    assert sat.bstar == 9.9999997e-05


def test_star():
    record = 'Sadr,f|S|F8,20:22:13.7|2.43,40:15:24|-0.93,2.23,2000,0'
    e = readdb(record)
    assert isinstance(e, FixedBody)
    assert e.name == 'Sadr'
    assert e.coord.ra.to_string(sep=':', unit='hour') == '20:22:13.7'
    assert e.coord.dec.to_string(sep=':', unit='deg') == '40:15:24'
