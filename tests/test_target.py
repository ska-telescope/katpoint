################################################################################
# Copyright (c) 2009-2011,2013-2023, National Research Foundation (SARAO)
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

"""Tests for the target module."""

import pickle
from contextlib import contextmanager

import astropy.units as u
import numpy as np
import pytest
from astropy import __version__ as astropy_version
from astropy.coordinates import Angle
from packaging.version import Version

import katpoint

from .helper import check_separation

TLE_TARGET = (
    "GPS BIIA-21 (PRN 09), tle, "
    "1 22700U 93042A   07266.32333151  .00000012  00000-0  10000-3 0  8054, "
    "2 22700  55.4408  61.3790 0191986  78.1802 283.9935  2.00561720104282"
)


def test_construct_target():
    """Test various ways to construct targets, also with overridden parameters."""
    t0 = katpoint.Target(
        "J1939-6342 | PKS 1934-63, radec bfcal, "
        "19:39:25.0, -63:42:45.6, (408.0 8640.0 -30.76 26.49 -7.098 0.6053)"
    )
    # Construct Target from Target
    assert katpoint.Target(t0) == t0
    # Construct Target from description string
    assert katpoint.Target(t0.description) == t0
    # Construct Target from Body
    assert (
        katpoint.Target(t0.body, t0.name, t0.user_tags, t0.aliases, t0.flux_model) == t0
    )
    # Override some parameters
    a0 = katpoint.Antenna(
        "XDM, -25:53:23.0, 27:41:03.0, 1406.1086, 15.0, 1 2 3, 1 2 3, 1.14"
    )
    t0b = katpoint.Target(
        t0.description,
        name="Marie",
        user_tags="no_collab",
        aliases=["Fee", "Fie", "Foe"],
        antenna=a0,
        flux_frequency=1284.0 * u.MHz,
    )
    assert t0b.body.coord == t0.body.coord
    assert t0b.name == "Marie"
    assert t0b.tags == ["radec", "no_collab"]
    assert t0b.user_tags == ["no_collab"]
    assert t0b.aliases == ("Fee", "Fie", "Foe")
    assert t0b.flux_model == t0.flux_model
    assert t0b.antenna == a0
    assert t0b.flux_frequency == 1.284 * u.GHz
    # Check that we can also replace non-default parameters with defaults
    t0c = katpoint.Target(
        t0, name="", flux_model=None, antenna=None, flux_frequency=None
    )
    assert t0c.body.coord == t0.body.coord
    assert t0c.name == t0.body.default_name  # Target name cannot be empty - use default
    assert t0c.tags == t0.tags
    assert t0c.aliases == t0.aliases
    assert t0c.flux_model is None
    assert t0c.antenna is None
    assert t0c.flux_frequency is None
    # Check that construction from Target is nearly exact (within 10 nanoarcsec)
    t1 = katpoint.Target.from_radec(np.e * u.deg, np.pi * u.deg)
    t2 = katpoint.Target(t1)
    check_separation(t2.body.coord, np.e * u.deg, np.pi * u.deg, tol=10 * u.narcsec)
    assert t1.name == t2.name
    assert t1.tags == t2.tags
    assert t1.aliases == t2.aliases
    assert t1.flux_model == t2.flux_model
    # Construct from Body and SkyCoord only
    assert katpoint.Target(t1.body) == t1
    assert katpoint.Target(t1.body.coord) == t1
    # Bytes are right out
    with pytest.raises(TypeError):
        katpoint.Target(b"azel, -30.0, 90.0")
    with pytest.raises(TypeError):
        katpoint.Target(t0, flux_frequency=1284.0)


def test_construct_target_from_azel():
    """Test construction of targets from (az, el) vs strings."""
    azel1 = katpoint.Target("azel, 10.0, -10.0")
    azel2 = katpoint.Target.from_azel("10:00:00.0", "-10:00:00.0")
    assert azel1 == azel2, "Special azel constructor failed"
    with pytest.warns(FutureWarning):
        azel2_deprecated = katpoint.construct_azel_target("10:00:00.0", "-10:00:00.0")
    assert azel1 == azel2_deprecated, "Deprecated azel constructor failed"


def test_construct_target_from_radec():
    """Test construction of targets from (ra, dec) vs strings."""
    radec1 = katpoint.Target("radec, 20.0, -20.0")
    radec2 = katpoint.Target.from_radec("20.0", "-20.0")
    assert radec1 == radec2, "Special radec constructor (decimal) failed"
    with pytest.warns(FutureWarning):
        radec2_deprecated = katpoint.construct_radec_target("20.0", "-20.0")
    assert radec1 == radec2_deprecated, "Deprecated radec constructor failed"
    radec3 = katpoint.Target("radec, 20:00:00, -20:00:00")
    radec4 = katpoint.Target.from_radec("20:00:00.0", "-20:00:00.0")
    assert radec3 == radec4, "Special radec constructor (sexagesimal) failed"
    radec5 = katpoint.Target.from_radec("20:00:00.0", "-00:30:00.0")
    radec6 = katpoint.Target.from_radec("300.0", "-0.5")
    assert (
        radec5 == radec6
    ), "Special radec constructor (decimal <-> sexagesimal) failed"


def test_constructed_coords():
    """Test whether calculated coordinates match those with which it is constructed."""
    # azel
    azel = katpoint.Target("azel, 10.0, -10.0")
    calc_azel = azel.azel()
    assert calc_azel.az.deg == 10.0
    assert calc_azel.alt.deg == -10.0
    # radec (degrees)
    radec = katpoint.Target("radec, 20.0, -20.0")
    calc_radec = radec.radec()
    assert calc_radec.ra.deg == 20.0
    assert calc_radec.dec.deg == -20.0
    # radec (hours)
    radec_rahours = katpoint.Target("radec, 20:00:00, -20:00:00")
    calc_radec_rahours = radec_rahours.radec()
    assert calc_radec_rahours.ra.hms == (20, 0, 0)
    assert calc_radec_rahours.dec.deg == -20.0
    # gal
    lb = katpoint.Target("gal, 30.0, -30.0")
    calc_lb = lb.galactic()
    assert calc_lb.l.deg == 30.0
    assert calc_lb.b.deg == -30.0


def test_compare_update_target():
    """Test various ways to compare and update targets."""
    # Check that description string updates when object is updated
    t1 = katpoint.Target("piet, azel, 20, 30")
    t2 = katpoint.Target("piet | bollie, azel, 20, 30")
    assert t1 != t2, "Targets should not be equal"
    t1 = katpoint.Target(t1, aliases=["bollie"])
    assert t1.description == t2.description, "Target description string not updated"
    assert t1 == t2.description, "Equality with description string failed"
    assert t1 == t2, "Equality with target failed"
    assert t1 == katpoint.Target(t2), "Construction with target object failed"
    assert t1 == pickle.loads(pickle.dumps(t1)), "Pickling failed"
    try:
        assert hash(t1) == hash(t2), "Target hashes not equal"
    except TypeError:
        pytest.fail("Target object not hashable")
    t2.add_tags("different")
    assert t1 != t2, "Targets should not be equal"


@pytest.mark.parametrize(
    "description",
    [
        "Venus | Flytrap | De Milo, azel, 10, 20",
        "Venus | Flytrap | De Milo, radec, 10, 20",
        "Venus | Flytrap | De Milo, gal, 10, 20",
        "Venus | Flytrap | De Milo, special",
        "Venus | Flytrap | De Milo" + TLE_TARGET[TLE_TARGET.find(",") :],
        (
            "xephem radec, "
            "Venus|Flytrap|De Milo~f|S|F8~20:22:13.7|2.43~40:15:24|-0.93~2.23~2000~0"
        ),
        (
            "Venus, xephem radec, "
            "Flytrap|De Milo~f|S|F8~20:22:13.7|2.43~40:15:24|-0.93~2.23~2000~0"
        ),
        (
            "xephem tle, Venus|Flytrap|De Milo"
            "~E~7/16.82966206/2016| 4/7.82812/2016|10/24.8281/2016"
            "~0.054400001~244.0062~8.9699999e-05~182.4502~200.0764"
            "~1.00273159~1.53e-06~1697~0"
        ),
    ],
)
def test_names(description):
    """Test that targets have appropriate names and aliases."""
    target = katpoint.Target(description)
    assert target.name == "Venus"
    assert target.aliases == ("Flytrap", "De Milo")
    # Names are read-only
    with pytest.raises(AttributeError):
        target.name = "bollie"
    with pytest.raises(AttributeError):
        target.aliases = ("bollie",)
    with pytest.raises(AttributeError):
        target.aliases += ("bollie",)
    new_aliases = ["Aphrodite"]
    override_target = katpoint.Target(description, aliases=new_aliases)
    # Check that we can't mutate the aliases after creating the Target
    new_aliases.append("Mighty")
    assert override_target.aliases == ("Aphrodite",)


def test_add_tags():
    """Test adding tags."""
    tag_target = katpoint.Target("azel J2000 GPS, 40.0, -30.0")
    tag_target.add_tags(None)
    tag_target.add_tags("pulsar")
    tag_target.add_tags(["SNR", "GPS SNR"])
    assert tag_target.tags == [
        "azel",
        "J2000",
        "GPS",
        "pulsar",
        "SNR",
    ], "Added tags not correct"
    assert tag_target.user_tags == [
        "J2000",
        "GPS",
        "pulsar",
        "SNR",
    ], "Added tags not correct"


FLUX_MODEL = katpoint.FluxDensityModel.from_description("(1000.0 2000.0 1.0 10.0)")


@pytest.mark.parametrize(
    "description",
    [
        f"azel, 20, 30, {FLUX_MODEL.description}",
        f"radec, -10, 20, {FLUX_MODEL.description}",
        f"gal, 10, 20, {FLUX_MODEL.description}",
        f"Sun, special, {FLUX_MODEL.description}",
        f"{TLE_TARGET}, {FLUX_MODEL.description}",
        (
            "xephem, Sadr~f|S|F8~20:22:13.7|2.43~40:15:24|-0.93~2.23~2000~0, "
            f"{FLUX_MODEL.description}"
        ),
    ],
)
def test_flux_model_on_all_targets(description):
    """Test that any target type can have a flux model."""
    target = katpoint.Target(description)
    assert target.flux_model == FLUX_MODEL


@pytest.mark.parametrize(
    "description",
    [
        "azel, -30.0, 90.0",
        "azel, -30.0d, 90.0d",
        ", azel, 180, -45:00:00.0",
        "Zenith, azel, 0, 90",
        "radec J2000, 0, 0.0, (1000.0 2000.0 1.0 10.0)",
        ", radec B1950, 14:23:45.6, -60:34:21.1",
        "radec B1900, 14:23:45.6, -60:34:21.1",
        "radec B1900, 14 23 45.6, -60 34 21.1",
        "radec B1900, 14:23:45.6h, -60:34:21.1d",
        "gal, 300.0, 0.0",
        "gal, 300.0d, 0.0d",
        "gal, 300.0, 0.0,,,,,,,,,,,,,,",
        "Sag A, gal, 0.0, 0.0",
        "Zizou, radec cal, 1.4, 30.0, (1000.0 2000.0 1.0 10.0)",
        "Fluffy | *Dinky, radec, 12.5, -50.0, (1.0 2.0 1.0 2.0 3.0 4.0)",
        TLE_TARGET,
        # Unnamed satellite
        TLE_TARGET[TLE_TARGET.find(",") :],
        "Sun, special",
        "Nothing, special",
        "Moon | Luna, special solarbody",
        "xephem radec, Sadr~f|S|F8~20:22:13.7|2.43~40:15:24|-0.93~2.23~2000~0",
        (
            "Acamar | Theta Eridani, xephem, "
            "HIC 13847~f|S|A4~2:58:16.03~-40:18:17.1~2.906~2000~0"
        ),
        (
            "Kakkab, xephem, "
            "H71860 | S225128~f|S|B1~14:41:55.768~-47:23:17.51~2.304~2000~0"
        ),
        (
            "xephem tle GEO, INTELSAT NEW DAWN"
            "~E~7/16.82966206/2016| 4/7.82812/2016|10/24.8281/2016"
            "~0.054400001~244.0062~8.9699999e-05~182.4502~200.0764"
            "~1.00273159~1.53e-06~1697~0"
        ),
        (
            "INTELSAT NEW DAWN, tle GEO, "
            "1 37392U 11016A   16198.82966206  .00000153  00000-0  00000-0 0  9996,"
            "2 37392   0.0544 244.0062 0000897 182.4502 200.0764  1.00273159 16973"
        ),
    ],
)
def test_construct_valid_target(description):
    """Test construction of valid targets from strings and vice versa."""
    # Normalise description string through one cycle to allow comparison
    reference_description = katpoint.Target(description).description
    test_target = katpoint.Target(reference_description)
    assert test_target.description == reference_description, (
        f"Target description ('{test_target.description}') "
        f"differs from reference ('{reference_description}')"
    )
    # Exercise repr() and str()
    print(f"{test_target!r} {test_target}")


@pytest.mark.parametrize(
    "description",
    [
        "",
        "Sun",
        "Sun, ",
        "-30.0, 90.0",
        ", azel, -45:00:00.0",
        "Zenith, azel blah",
        "radec J2000, 0.3",
        "gal, 0.0",
        "Zizou, radec cal, 1.4, 30.0, (1000.0, 2000.0, 1.0, 10.0)",
        # The old TLE format containing three newline-separated lines
        # straight from TLE file.
        (
            "tle, GPS BIIA-21 (PRN 09)    \n"
            "1 22700U 93042A   07266.32333151  .00000012  00000-0  10000-3 0  8054\n"
            "2 22700  55.4408  61.3790 0191986  78.1802 283.9935  2.00561720104282\n"
        ),
        # TLE missing the first line
        (
            "GPS BIIA-21 (PRN 09), tle, "
            "2 22700  55.4408  61.3790 0191986  78.1802 283.9935  2.00561720104282"
        ),
        # TLE missing the satellite catalog number and classification on line 1
        (
            "GPS BIIA-22 (PRN 05), tle, "
            "1 93054A   07266.92814765  .00000062  00000-0  10000-3 0  2895, "
            "2 22779  53.8943 118.4708 0081407  68.2645 292.7207  2.00558015103055"
        ),
        "Sunny, special",
        "Slinky, star",
        "Sadr, xephem",
        "xephem star, Sadr~20:22:13.7|2.43~40:15:24|-0.93~2.23~2000~0",
        "hotbody, 34.0, 45.0",
        "The dreaded em dash, radec, 12:34:56h, \N{em dash}60:43:21d",
    ],
)
def test_construct_invalid_target(description):
    """Test construction of invalid targets from strings."""
    with pytest.raises(ValueError):
        katpoint.Target(description)


@pytest.mark.parametrize(
    "description",
    [
        # Basic no-name coordinates
        "azel, 20:00:00d, 30:00:00d",
        "radec, 12:00:00h, 24:00:00d",
        "gal, 300d, 0d",
        # The name lives in the EDB string instead
        (
            "xephem tle, INTELSAT NEW DAWN | FUNKY"
            "~E~7/16.82966206/2016| 4/7.82812/2016|10/24.8281/2016"
            "~0.054400001~244.0062~8.9699999e-05~182.4502~200.0764"
            "~1.00273159~1.53e-06~1697~0"
        ),
        # No-name TLE
        (
            "xephem tle, "
            "~E~7/16.82966206/2016| 4/7.82812/2016|10/24.8281/2016"
            "~0.054400001~244.0062~8.9699999e-05~182.4502~200.0764"
            "~1.00273159~1.53e-06~1697~0"
        ),
    ],
)
def test_no_name_targets(description):
    """Test that targets of any type can be anonymous."""
    assert katpoint.Target(description).description == description


NON_AZEL = "astrometric_radec apparent_radec galactic"


@contextmanager
def _does_not_raise(error):  # pylint: disable=unused-argument
    yield


@pytest.mark.parametrize(
    "description,methods,raises,error",
    [
        ("azel, 10, -10", "azel", _does_not_raise, None),
        ("azel, 10, -10", NON_AZEL, pytest.raises, ValueError),
        ("radec, 20, -20", "azel", pytest.raises, ValueError),
        ("radec, 20, -20", NON_AZEL, _does_not_raise, None),
        ("gal, 30, -30", "azel", pytest.raises, ValueError),
        ("gal, 30, -30", NON_AZEL, _does_not_raise, None),
        ("Sun, special", "azel", pytest.raises, ValueError),
        ("Sun, special", NON_AZEL, _does_not_raise, None),
        (TLE_TARGET, "azel", pytest.raises, ValueError),
        (TLE_TARGET, NON_AZEL, _does_not_raise, None),
    ],
)
def test_coord_methods_without_antenna(description, methods, raises, error):
    """Test whether coordinate methods can operate without an Antenna."""
    target = katpoint.Target(description)
    for method in methods.split():
        with raises(error):
            getattr(target, method)()


TARGET = katpoint.Target.from_azel("45:00:00.0", "75:00:00.0")
ANT1 = katpoint.Antenna("A1, -31.0, 18.0, 0.0, 12.0, 0.0 0.0 0.0")
ANT2 = katpoint.Antenna("A2, -31.0, 18.0, 0.0, 12.0, 10.0 -10.0 0.0")
TS = katpoint.Timestamp("2013-08-14 09:25")


def _array_vs_scalar(func, array_in, sky_coord=False, pre_shape=(), post_shape=()):
    """Check that `func` output for ndarray of inputs is array of scalar outputs."""
    array_out = func(array_in)
    # XXX Workaround for Astropy 4.2 regression (np.shape used to work, now TypeError)
    try:
        out_shape = np.shape(array_out)
    except TypeError:
        out_shape = array_out.shape
    assert out_shape == pre_shape + array_in.shape + post_shape
    all_pre = len(pre_shape) * (np.s_[:],)
    all_post = len(post_shape) * (np.s_[:],)
    for index_in in np.ndindex(array_in.shape):
        scalar = func(array_in[index_in])
        if sky_coord:
            # Treat output as if it is SkyCoord with internal array,
            # check separation instead.
            assert array_out[index_in].separation(scalar).rad == pytest.approx(
                0.0, abs=2e-12
            )
        else:
            # Assume that function outputs more complicated ndarrays of numbers
            # (or equivalent)
            array_slice = np.asarray(array_out)[all_pre + index_in + all_post]
            np.testing.assert_array_equal(array_slice, np.asarray(scalar))


@pytest.mark.parametrize(
    "description",
    ["azel, 10, -10", "radec, 20, -20", "gal, 30, -30", "Sun, special", TLE_TARGET],
)
def test_array_valued_methods(description):
    """Test array-valued methods, comparing output against scalar versions."""
    offsets = np.array([[[0, 1], [4, 5]]])
    times = (katpoint.Timestamp("2020-07-30 14:02:00") + offsets).time
    assert times.shape == offsets.shape
    target = katpoint.Target(description)
    _array_vs_scalar(lambda t: target.azel(t, ANT1), times, sky_coord=True)
    _array_vs_scalar(lambda t: target.apparent_radec(t, ANT1), times, sky_coord=True)
    _array_vs_scalar(lambda t: target.astrometric_radec(t, ANT1), times, sky_coord=True)
    _array_vs_scalar(lambda t: target.galactic(t, ANT1), times, sky_coord=True)
    _array_vs_scalar(lambda t: target.parallactic_angle(t, ANT1), times)
    _array_vs_scalar(lambda t: target.geometric_delay(ANT2, t, ANT1)[0], times)
    _array_vs_scalar(lambda t: target.geometric_delay(ANT2, t, ANT1)[1], times)
    _array_vs_scalar(lambda t: target.uvw_basis(t, ANT1), times, pre_shape=(3, 3))
    _array_vs_scalar(
        lambda t: target.uvw([ANT1, ANT2], t, ANT1), times, post_shape=(2,)
    )
    _array_vs_scalar(lambda t: target.lmn(0.0, 0.0, t, ANT1), times, pre_shape=(3,))
    l, m, n = target.lmn(np.zeros_like(offsets), np.zeros_like(offsets), times, ANT1)
    assert l.shape == m.shape == n.shape == offsets.shape
    assert np.allclose(
        target.separation(target, times, ANT1).rad, np.zeros_like(offsets), atol=2e-12
    )


def test_coords():
    """Test coordinate conversions for coverage and verification."""
    coord = TARGET.azel(TS, ANT1)
    assert coord.az.deg == 45  # PyEphem: 45
    assert coord.alt.deg == 75  # PyEphem: 75
    coord = TARGET.apparent_radec(TS, ANT1)
    check_separation(coord, "8:53:03.49166920h", "-19:54:51.92328722d", tol=1 * u.mas)
    # PyEphem:               8:53:09.60,          -19:51:43.0 (same as astrometric)
    coord = TARGET.astrometric_radec(TS, ANT1)
    check_separation(coord, "8:53:09.60397465h", "-19:51:42.87773802d", tol=1 * u.mas)
    # PyEphem:               8:53:09.60,          -19:51:43.0
    coord = TARGET.galactic(TS, ANT1)
    check_separation(coord, "245:34:49.20442837d", "15:36:24.87974969d", tol=1 * u.mas)
    # PyEphem:               245:34:49.3,           15:36:24.7
    coord = TARGET.parallactic_angle(TS, ANT1)
    assert coord.deg == pytest.approx(-140.279593566336)  # PyEphem: -140.34440985011398


DELAY_TARGET = katpoint.Target("radec, 20.0, -20.0")
DELAY_TS = [TS, TS + 1.0]
DELAY = [1.75538294e-08, 1.75522002e-08] * u.s
DELAY_RATE = [-1.62915174e-12, -1.62929689e-12] * (u.s / u.s)
UVW = (
    [-7.118580813334029, -11.028682662045913, -5.262505671628351],
    [-7.119215642091996, -11.028505936045280, -5.262017242465739],
) * u.m


def test_delay():
    """Test geometric delay."""
    delay, delay_rate = DELAY_TARGET.geometric_delay(ANT2, DELAY_TS[0], ANT1)
    assert np.allclose(delay, DELAY[0], rtol=0, atol=0.001 * u.ps)
    assert np.allclose(delay_rate, DELAY_RATE[0], rtol=1e-10, atol=1e-20)
    delay, delay_rate = DELAY_TARGET.geometric_delay(ANT2, DELAY_TS, ANT1)
    assert np.allclose(delay, DELAY, rtol=0, atol=0.001 * u.ps)
    assert np.allclose(delay_rate, DELAY_RATE, rtol=1e-10, atol=1e-20)


def test_uvw():
    """Test uvw calculation."""
    uvw = DELAY_TARGET.uvw(ANT2, DELAY_TS[0], ANT1)
    assert np.allclose(uvw.xyz, UVW[0], rtol=0, atol=10 * u.nm)
    uvw = DELAY_TARGET.uvw(ANT2, DELAY_TS, ANT1)
    assert np.allclose(uvw.xyz, UVW.T, rtol=0, atol=10 * u.nm)


def test_uvw_timestamp_array_azel():
    """Test uvw calculation on a timestamp array when the target is an azel target."""
    azel = DELAY_TARGET.azel(DELAY_TS[0], ANT1)
    target = katpoint.Target.from_azel(azel.az, azel.alt)
    uvw = target.uvw(ANT2, DELAY_TS, ANT1)
    assert np.allclose(uvw[0].xyz, UVW[0], rtol=0, atol=10 * u.nm)
    assert np.allclose(uvw.z, [UVW[0, 2]] * len(DELAY_TS), rtol=0, atol=10 * u.nm)


def test_uvw_antenna_array():
    """Test that uvw can be computed on a list of baselines to speed things up."""
    uvw = DELAY_TARGET.uvw([ANT1, ANT2], DELAY_TS[0], ANT1)
    assert np.allclose(uvw.xyz, np.c_[np.zeros(3), UVW[0]], rtol=0, atol=10 * u.nm)


def test_uvw_both_array():
    """Test that uvw can be computed on a list of baselines and times at once."""
    uvw = DELAY_TARGET.uvw([ANT1, ANT2], DELAY_TS, ANT1)
    # UVW array has shape (3, n_times, n_bls)
    # stack times along dim 1 and ants along dim 2
    desired_uvw = np.dstack([np.zeros((3, len(DELAY_TS))), UVW.T])
    assert np.allclose(uvw.xyz, desired_uvw, rtol=0, atol=10 * u.nm)


def test_uvw_hemispheres():
    """Test uvw calculation near the equator.

    The implementation behaves differently depending on the sign of
    declination. This test is to catch sign flip errors.
    """
    target1 = katpoint.Target.from_radec(0 * u.deg, -0.2 * u.mas)
    target2 = katpoint.Target.from_radec(0 * u.deg, +0.2 * u.mas)
    uvw1 = target1.uvw(ANT2, TS, ANT1)
    uvw2 = target2.uvw(ANT2, TS, ANT1)
    assert np.allclose(uvw1.xyz, uvw2.xyz, rtol=0, atol=25 * u.micron)


def test_lmn():
    """Test lmn calculation."""
    # For angles less than pi/2, it matches SIN projection
    pointing = katpoint.Target.from_radec("11:00:00.0", "-75:00:00.0")
    target = katpoint.Target.from_radec("16:00:00.0", "-65:00:00.0")
    radec = target.radec(timestamp=TS, antenna=ANT1)
    l, m, n = pointing.lmn(radec.ra.rad, radec.dec.rad)
    expected_l, expected_m = pointing.sphere_to_plane(
        radec.ra.rad, radec.dec.rad, projection_type="SIN", coord_system="radec"
    )
    expected_n = np.sqrt(1.0 - expected_l**2 - expected_m**2)
    np.testing.assert_almost_equal(l, expected_l, decimal=12)
    np.testing.assert_almost_equal(m, expected_m, decimal=12)
    np.testing.assert_almost_equal(n, expected_n, decimal=12)
    # Test angle > pi/2: using the diametrically opposite target
    l, m, n = pointing.lmn(np.pi + radec.ra.rad, -radec.dec.rad)
    np.testing.assert_almost_equal(l, -expected_l, decimal=12)
    np.testing.assert_almost_equal(m, -expected_m, decimal=12)
    np.testing.assert_almost_equal(n, -expected_n, decimal=12)


def test_separation():
    """Test separation calculation."""
    sun = katpoint.Target("Sun, special")
    azel_sun = sun.azel(TS, ANT1)
    azel = katpoint.Target.from_azel(azel_sun.az, azel_sun.alt)
    sep = sun.separation(azel, TS, ANT1)
    np.testing.assert_almost_equal(sep.rad, 0.0)
    sep = azel.separation(sun, TS, ANT1)
    np.testing.assert_almost_equal(sep.rad, 0.0)
    azel2 = katpoint.Target.from_azel(
        azel_sun.az, azel_sun.alt + Angle(0.01, unit=u.rad)
    )
    sep = azel.separation(azel2, TS, ANT1)
    np.testing.assert_almost_equal(sep.rad, 0.01, decimal=12)
    # Check that different default antennas are handled correctly
    azel.antenna = ANT1
    sun.antenna = ANT2
    sep = azel.separation(sun, TS)
    np.testing.assert_almost_equal(sep.rad, 0.0)


def test_projection():
    """Test projection."""
    az, el = np.radians(50.0), np.radians(80.0)
    x, y = TARGET.sphere_to_plane(az, el, TS, ANT1)
    re_az, re_el = TARGET.plane_to_sphere(x, y, TS, ANT1)
    np.testing.assert_almost_equal(re_az, az, decimal=12)
    np.testing.assert_almost_equal(re_el, el, decimal=12)


def _ant_vs_location(func, atol=0.0):
    """Check that `func(ant1, ant2)` output is same for Antennas and EarthLocations."""
    ant_output = func(ANT1, ANT2)
    location_output = func(ANT1.location, ANT2.location)
    try:
        # Use sky coordinate separation to obtain floating-point difference
        separation = location_output.separation(ant_output)
        assert np.allclose(separation, 0.0, rtol=0.0, atol=atol)
    except AttributeError:
        try:
            assert np.allclose(location_output.xyz, ant_output.xyz, rtol=0.0, atol=atol)
        except AttributeError:
            assert np.allclose(location_output, ant_output, rtol=0.0, atol=atol)


def test_earth_location():
    """Test that Antenna parameters accept EarthLocations."""
    offsets = np.array([[[0, 1, 2, 3], [4, 5, 6, 7]]])
    timestamps = katpoint.Timestamp("2021-02-11 14:28:00") + offsets
    target = katpoint.Target("radec, 20, -20")
    _ant_vs_location(lambda a1, a2: target.azel(timestamps, a1))
    _ant_vs_location(lambda a1, a2: target.apparent_radec(timestamps, a1))
    _ant_vs_location(lambda a1, a2: target.astrometric_radec(timestamps, a1))
    _ant_vs_location(lambda a1, a2: target.galactic(timestamps, a1))
    _ant_vs_location(lambda a1, a2: target.parallactic_angle(timestamps, a1))
    _ant_vs_location(
        lambda a1, a2: target.geometric_delay(a2, timestamps, a1)[0], atol=1e-16
    )
    _ant_vs_location(
        lambda a1, a2: target.geometric_delay(a2, timestamps, a1)[1], atol=1e-21
    )
    _ant_vs_location(lambda a1, a2: target.uvw_basis(timestamps, a1))
    _ant_vs_location(lambda a1, a2: target.uvw([a1, a2], timestamps, a1), atol=1e-9)
    _ant_vs_location(
        lambda a1, a2: target.uvw(np.stack([a1, a2]), timestamps, a1), atol=1e-9
    )
    _ant_vs_location(lambda a1, a2: target.lmn(0.0, 0.0, timestamps, a1))
    _ant_vs_location(lambda a1, a2: target.separation(target, timestamps, a1))
    _ant_vs_location(lambda a1, a2: target.plane_to_sphere(0.1, 0.1, timestamps, a1)[0])
    _ant_vs_location(lambda a1, a2: target.plane_to_sphere(0.1, 0.1, timestamps, a1)[1])
    _ant_vs_location(lambda a1, a2: target.sphere_to_plane(0.1, 0.1, timestamps, a1)[0])
    _ant_vs_location(lambda a1, a2: target.sphere_to_plane(0.1, 0.1, timestamps, a1)[1])


# TLE for ISS on 2020-12-17
ISS = katpoint.Target(
    "ISS (ZARYA), tle,"
    "1 25544U 98067A   20351.71912775  .00000900  00000-0  24328-4 0  9992,"
    "2 25544  51.6442 165.2978 0001589 133.0028 320.9621 15.49190988260311"
)
astropy_version = Version(astropy_version)


def test_great_conjunction():
    """Use the Great Conjunction to test astrometric (ra, dec) for different bodies."""
    # Recreate Jason de Freitas's observation of the ISS
    # passing between Jupiter and Saturn, based on
    # https://petapixel.com/2020/12/22/photographer-captures-iss-passing-between-jupiter-and-saturn/
    # The altitude is above sea level instead of WGS84, but should be close enough.
    pentax = katpoint.Antenna("Jellore Lookout NSW, -34.462653, 150.427971, 864")
    # The photo was taken "at around 9:54pm". Australian Eastern Daylight Time (AEDT)
    # is 11 hours ahead of UTC => therefore around 10:54 UTC
    timestamp = katpoint.Timestamp("2020-12-17 10:53:10")
    jupiter = katpoint.Target("Jupiter, special")
    saturn = katpoint.Target("Saturn, special")
    moon = katpoint.Target("Moon, special")
    j = jupiter.radec(timestamp, pentax)
    s = saturn.radec(timestamp, pentax)
    i = ISS.radec(timestamp, pentax)
    m = moon.radec(timestamp, pentax)
    # This is a regression test, using separations as measured by Astropy 4.3
    # (also valid for 4.1)
    # The accuracy is within the precision (9 digits => 0.5e-9 deg = 1.8 microarcsec)
    assert np.allclose(j.separation(s), 0.486585894 * u.deg, atol=0.0018 * u.mas)
    assert np.allclose(j.separation(i), 0.213263690 * u.deg, atol=0.0018 * u.mas)
    assert np.allclose(i.separation(s), 0.275048635 * u.deg, atol=0.0018 * u.mas)
    # The Moon model improved in Astropy 5.0
    tol = 0.0018 * u.mas if astropy_version >= Version("5.0") else 300 * u.mas
    assert np.allclose(m.separation(i), 3.262362502 * u.deg, atol=tol)


def test_improved_azel():
    """Improved (az,el) for nearby objects due to topocentric CIRS in Astropy 4.3."""
    # Check a more extreme case where the ISS is close to the observer (433 km)
    timestamp = katpoint.Timestamp("2020-12-15 17:25:59")
    pentax = katpoint.Antenna("Jellore Lookout NSW, -34.462653, 150.427971, 864")
    azel = ISS.azel(timestamp, pentax)
    # Check against Astropy 5.3 and relax tolerance for older versions
    tol = 0.1 * u.mas if astropy_version >= Version("5.3") else 4 * u.arcmin
    check_separation(azel, "137:54:59.0569d", "83:25:13.3345d", tol=tol)
