################################################################################
# Copyright (c) 2009-2011,2014-2021,2023, National Research Foundation (SARAO)
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

"""Tests for the antenna module."""

import pickle
import time

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import EarthLocation

import katpoint

from .helper import assert_angles_almost_equal


@pytest.mark.parametrize(
    "description",
    [
        "XDM, -25:53:23.0, 27:41:03.0, 1406.1086, 15.0",
        "FF1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 18.4 -8.7 0.0",
        (
            "FF2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 86.2 25.5 0.0, "
            "-0:06:39.6 0 0 0 0 0 0:09:48.9, 1.16"
        ),
        ", -25:53:23.0, 27:41:03.0, 1406.1086, 15.0",  # unnamed antenna
    ],
)
def test_construct_valid_antenna(description):
    """Test construction of valid antennas from strings and vice versa."""
    # Normalise description string through one cycle to allow comparison
    reference_description = katpoint.Antenna(description).description
    test_antenna = katpoint.Antenna(reference_description)
    assert (
        str(test_antenna) == test_antenna.description == reference_description
    ), "Antenna description differs from original string"


@pytest.mark.parametrize(
    "description",
    [
        "",
        "XDM, -25:53:23.05075, 27:41:03.0",
        "The dreaded em dash, \N{em dash}25:53:23.0, 27:41:03.0, 1406.1086, 15.0",
    ],
)
def test_construct_invalid_antenna(description):
    """Test construction of invalid antennas from strings."""
    with pytest.raises(ValueError):
        katpoint.Antenna(description)


def test_construct_antenna():
    """Test various ways to construct antennas, also with overridden parameters."""
    a0 = katpoint.Antenna(
        "XDM, -25:53:23.0, 27:41:03.0, 1406.1086, 15.0, 1 2 3, 1 2 3, 1.14"
    )
    # Construct Antenna from Antenna
    assert katpoint.Antenna(a0) == a0
    # Construct Antenna from description string
    assert katpoint.Antenna(a0.description) == a0
    # Construct Antenna from EarthLocation
    fields = a0.description.split(", ")
    name = fields[0]
    location = EarthLocation.from_geodetic(
        lat=fields[1], lon=fields[2], height=fields[3]
    )
    assert katpoint.Antenna(location, name, *fields[4:]).description == a0.description
    with pytest.raises(ValueError):
        katpoint.Antenna(location, name + ", oops", *fields[4:])
    # Exercise repr() and str()
    print(f"{a0!r} {a0}")
    # Override some parameters
    a0b = katpoint.Antenna(a0.description, name="bloop", beamwidth=np.pi)
    assert a0b.location == a0.location
    assert a0b.name == "bloop"
    assert a0b.diameter == a0.diameter
    assert a0b.delay_model == a0.delay_model
    assert a0b.pointing_model == a0.pointing_model
    assert a0b.beamwidth == np.pi
    # Check that we can also replace non-default parameters with defaults
    a0c = katpoint.Antenna(
        a0, name="", diameter=0.0, delay_model=None, pointing_model=None
    )
    assert a0c.location == a0.ref_location
    assert a0c.name == ""
    assert a0c.diameter == 0.0
    assert not a0c.delay_model
    assert not a0c.pointing_model
    assert a0c.beamwidth == a0.beamwidth
    # Check that construction from Antenna is exact
    location = EarthLocation.from_geodetic(lat=np.pi, lon=np.pi, height=np.e)
    a1 = katpoint.Antenna(location, name="pangolin", diameter=np.e, beamwidth=np.pi)
    a2 = katpoint.Antenna(a1)
    assert a1.location == a2.location == location
    assert a1.name == a2.name
    assert a1.diameter == a2.diameter
    assert a1.beamwidth == a2.beamwidth


def test_compare_update_antenna():
    """Test various ways to compare and update antennas."""
    a1 = katpoint.Antenna("FF1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 18.4 -8.7 0.0")
    a2 = katpoint.Antenna(
        "FF2, -30:43:17.3, 21:24:38.5, 1038.0, 13.0, 18.4 -8.7 0.0, 0.1, 1.22"
    )
    assert a1 != a2, "Antennas should be inequal"
    assert a1 < a2, "Antenna a1 should come before a2 when sorted by description string"
    assert (
        a1 <= a2
    ), "Antenna a1 should come before a2 when sorted by description string"
    assert a2 > a1, "Antenna a2 should come after a1 when sorted by description string"
    assert a2 >= a1, "Antenna a2 should come after a1 when sorted by description string"
    # Check that description string updates when object is updated
    a1.name = "FF2"
    a1.diameter = 13.0 * u.m
    a1.pointing_model = katpoint.PointingModel("0.1")
    a1.beamwidth = 1.22
    assert a1.description == a2.description, "Antenna description string not updated"
    assert a1 == a2.description, "Antenna not equal to description string"
    assert a1 == a2, "Antennas not equal"
    assert a1 == katpoint.Antenna(a2), "Construction with antenna object failed"
    assert a1 == pickle.loads(pickle.dumps(a1)), "Pickling failed"
    try:
        assert hash(a1) == hash(a2), "Antenna hashes not equal"
    except TypeError:
        pytest.fail("Antenna object not hashable")


def test_coordinates():
    """Test coordinates associated with antenna location."""
    lla = ("-30:42:39.8", "21:26:38.0", "1086.6")
    enu = (-8.264, -207.29, 8.5965)
    ant = katpoint.Antenna(
        f"m000, {', '.join(lla)}, 13.5, {' '.join(str(c) for c in enu)}"
    )
    ref_location = EarthLocation.from_geodetic(lat=lla[0], lon=lla[1], height=lla[2])
    assert ant.ref_location == ref_location
    assert ant.position_enu == enu
    ant0 = ant.array_reference_antenna()
    assert ant0.location == ref_location
    assert ant0.position_ecef == tuple(ref_location.itrs.cartesian.xyz.to_value(u.m))
    assert ant0.position_wgs84 == (
        ref_location.lat.to_value(u.rad),
        ref_location.lon.to_value(u.rad),
        ref_location.height.to_value(u.m),
    )
    assert np.array_equal(ant0.baseline_toward(ant).xyz, enu * u.m)
    reverse_bl = ant.baseline_toward(ant0)
    assert np.allclose(reverse_bl.xyz[:2], -(enu * u.m)[:2], rtol=0, atol=0.5 * u.mm)
    assert np.allclose(reverse_bl.xyz[2], -(enu * u.m)[2], rtol=0, atol=10 * u.mm)


def test_local_sidereal_time():
    """Test sidereal time and the use of date/time strings vs floats as timestamps."""
    ant = katpoint.Antenna("XDM, -25:53:23.0, 27:41:03.0, 1406.1086, 15.0")
    timestamp = "2009/07/07 08:36:20"
    utc_secs = (
        time.mktime(time.strptime(timestamp, "%Y/%m/%d %H:%M:%S")) - time.timezone
    )
    sid1 = ant.local_sidereal_time(timestamp)
    sid2 = ant.local_sidereal_time(utc_secs)
    assert sid1 == sid2, "Sidereal time differs for float and date/time string"
    sid3 = ant.local_sidereal_time([timestamp, timestamp])
    sid4 = ant.local_sidereal_time([utc_secs, utc_secs])
    assert_angles_almost_equal(
        np.array([a.rad for a in sid3]), np.array([a.rad for a in sid4]), decimal=12
    )


def test_array_reference_antenna():
    """Test that the reference antenna is correctly reported."""
    ant = katpoint.Antenna(
        "FF2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 86.2 25.5 0.0, "
        "-0:06:39.6 0 0 0 0 0 0:09:48.9, 1.16"
    )
    ref_ant = ant.array_reference_antenna()
    assert ref_ant.description == "array, -30:43:17.3d, 21:24:38.5d, 1038, 0, , , 1.22"


@pytest.mark.parametrize(
    "description",
    [
        "FF2, -30:43:17.3d, 21:24:38.5d, 1038, 12, 86.2 25.5, , 1.22",
        "FF2, -30:43:17.34567d, 21:24:38.56723d, 1038.1086, 12, 86.2 25.5, , 1.22",
        (
            "FF2, -30:43:17.12345678d, 21:24:38.12345678d, 1038.123456, "
            "12, 86.2 25.5, , 1.22"
        ),
        "FF2, -30:43:17.3d, 21:24:38.5d, 1038, 12, 86.123456 25.123456, , 1.22",
    ],
)
def test_description_round_trip(description):
    """Test that the description strings can round-trip for various precisions."""
    assert katpoint.Antenna(description).description == description


@pytest.mark.parametrize(
    "location",
    [
        # The canonical MeerKAT array centre in ITRS to nearest millimetre
        EarthLocation.from_geocentric(5109360.133, 2006852.586, -3238948.127, unit=u.m),
        # The canonical MeerKAT array centre in WGS84
        EarthLocation.from_geodetic("-30:42:39.8", "21:26:38.0", "1086.6"),
        # The WGS84 array centre in XYZ format (0.5 mm difference...)
        EarthLocation.from_geocentric(
            5109360.13332123, 2006852.58604291, -3238948.12747888, unit=u.m
        ),
        # Check location based on different ellipsoid (2 m difference from WGS84)
        EarthLocation.from_geodetic(
            "-30:42:39.8", "21:26:38.0", "1086.6", ellipsoid="WGS72"
        ),
    ],
)
def test_location_round_trip(location):
    """Test that locations can round-trip via Antenna description strings and back."""
    xyz = location.itrs.cartesian
    descr = katpoint.Antenna(location).description
    xyz2 = katpoint.Antenna(descr).location.itrs.cartesian
    assert (xyz2 - xyz).norm() < 1 * u.micron
