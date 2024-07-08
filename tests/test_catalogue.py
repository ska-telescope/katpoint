################################################################################
# Copyright (c) 2009-2011,2013-2021,2023, National Research Foundation (SARAO)
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

"""Tests for the catalogue module."""

import astropy.units as u
import numpy as np
import pytest

import katpoint

TARGETS = [
    "xephem radec, Acamar~f|S|A4~02 58 15.7156|-53.53~-40 18 17.046|25.71~2.88~2000~0",
    (
        "xephem radec, "
        "Alpheratz~f|S|B9~00 08 23.1680|135.68~+29 05 26.981|-162.95~2.07~2000~0"
    ),
    "xephem radec, Fomalhaut~f|S|A3~22:57:38.8|329.22~-29:37:19|-164.22~1.17~2000~0",
    (
        "xephem radec, "
        "Miaplacidus~f|S|A2~09 13 12.2408|-157.66~-69 43 02.901|108.91~1.67~2000~0"
    ),
    "xephem radec, Polaris~f|S|F7~2:31:47.1|44.22~89:15:51|-11.74~1.97~2000~0",
    "Sun, special",
    "Moon, special",
    "Jupiter, special",
    "Saturn, special",
    "Venus, special",
    "Zenith, azel, 0, 90",
]

FLUX_TARGET = katpoint.Target("flux, radec, 0.0, 0.0, (1.0 2.0 2.0 0.0 0.0)")
ANTENNA = katpoint.Antenna("XDM, -25:53:23.05075, 27:41:03.36453, 1406.1086, 15.0")
TIMESTAMP = "2009/06/14 12:34:56"


def test_catalogue_basic():
    """Basic catalogue tests."""
    cat = katpoint.Catalogue(TARGETS)
    repr(cat)
    str(cat)
    cat.add("# Comments will be ignored")
    with pytest.raises(ValueError):
        cat.add([1])


def test_catalogue_tab_completion():
    """Test that IPython tab completion will work."""
    cat = katpoint.Catalogue()
    cat.add("Nothing, special")
    cat.add("Earth | Terra Incognita, azel, 0, 0")
    cat.add("Earth | Sky, azel, 0, 90")
    # Check that it returns a sorted list
    assert cat._ipython_key_completions_() == [  # pylint: disable=protected-access
        "Earth",
        "Nothing",
        "Sky",
        "Terra Incognita",
    ]


def test_catalogue_same_name():
    """Test add() and remove() of targets with the same name."""
    cat = katpoint.Catalogue()
    targets = ["Sun, special", "Sun | Sol, special", "Sun, special hot"]
    # Add various targets called Sun
    cat.add(targets[0])
    assert cat["Sun"].description == targets[0]
    cat.add(targets[0])
    assert len(cat) == 1, "Did not ignore duplicate target"
    cat.add(targets[1])
    assert cat["Sun"].description == targets[1]
    cat.add(targets[2])
    assert cat["Sun"].description == targets[2]
    # Check length, iteration, membership
    assert len(cat) == len(targets)
    for n, t in enumerate(cat):
        assert t.description == targets[n]
    assert "Sun" in cat
    assert "Sol" in cat
    for t in targets:
        assert katpoint.Target(t) in cat
    # Remove targets one by one
    cat.remove("Sun")
    assert cat["Sun"].description == targets[1]
    cat.remove("Sun")
    assert cat["Sun"].description == targets[0]
    cat.remove("Sun")
    assert len(cat) == len(cat.targets) == len(cat.lookup) == 0, "Catalogue not empty"


def test_construct_catalogue(caplog):
    """Test construction of catalogues."""
    cat = katpoint.Catalogue(TARGETS, antenna=ANTENNA)
    num_targets_original = len(cat)
    assert num_targets_original == len(TARGETS)
    # Add target already in catalogue - no action
    cat.add(katpoint.Target("Sun, special"))
    num_targets = len(cat)
    assert num_targets == num_targets_original, "Number of targets incorrect"
    cat2 = katpoint.Catalogue(TARGETS)
    cat2.add(katpoint.Target("Sun, special"))
    assert cat == cat2, "Catalogues not equal"
    try:
        assert hash(cat) == hash(cat2), "Catalogue hashes not equal"
    except TypeError:
        pytest.fail("Catalogue object not hashable")
    # Add different targets with the same name
    cat2.add(katpoint.Target("Sun, special hot"))
    cat2.add(katpoint.Target("Sun | Sol, special"))
    assert len(cat2) == num_targets_original + 2, "Number of targets incorrect"
    cat2.remove("Sol")
    assert len(cat2) == num_targets_original + 1, "Number of targets incorrect"
    assert cat != cat2, "Catalogues should not be equal"
    test_target = cat.targets[-1]
    assert test_target.description == cat[test_target.name].description, "Lookup failed"
    assert cat["Non-existent"] is None, "Lookup of non-existent target failed"
    tle_lines = [
        "# Near-Earth object (comment ignored)\n",
        "IRIDIUM 7 [-]           \n",
        "1 24793U 97020B   19215.43137162  .00000054  00000-0  12282-4 0  9996\n",
        "2 24793  86.3987 354.1155 0002085  89.4941 270.6494 14.34287341164608\n",
        "# Deep-space object\n",
        "GPS BIIA-21 (PRN 09)    \n",
        "1 22700U 93042A   07266.32333151  .00000012  00000-0  10000-3 0  8054\n",
        "2 22700  55.4408  61.3790 0191986  78.1802 283.9935  2.00561720104282\n",
    ]
    cat.add_tle(tle_lines, "tle")
    assert "2 of 2 TLE set(s) are outdated" in caplog.text, "TLE epoch checks failed"
    assert (
        "deep-space" in caplog.text
    ), "Worst TLE epoch should be deep-space GPS satellite"
    edb_lines = [
        "# Comment ignored\n",
        "HIC 13847,f|S|A4,2:58:16.03,-40:18:17.1,2.906,2000,\n",
    ]
    cat.add_edb(edb_lines, "edb")
    assert len(cat.targets) == num_targets + 3, "Number of targets incorrect"
    cat.remove(cat.targets[-1].name)
    assert len(cat.targets) == num_targets + 2, "Number of targets incorrect"
    closest_target, dist = cat.closest_to(test_target)
    assert (
        closest_target.description == test_target.description
    ), "Closest target incorrect"
    assert np.allclose(
        dist, 0.0, rtol=0.0, atol=0.5e-5 * u.deg
    ), "Target should be on top of itself"


def test_that_equality_and_hash_ignore_order():
    """Test that shuffled catalogues are equal and produce the same hash."""
    a = katpoint.Catalogue()
    b = katpoint.Catalogue()
    t1 = katpoint.Target("Nothing, special")
    t2 = katpoint.Target("Sun, special")
    a.add(t1)
    a.add(t2)
    b.add(t2)
    b.add(t1)
    assert a == b, "Shuffled catalogues are not equal"
    assert hash(a) == hash(b), "Shuffled catalogues have different hashes"


def test_skip_empty():
    """Test that Catalogue can handle empty files without crashing."""
    cat = katpoint.Catalogue(["", "# comment", "   ", "\t\r "])
    assert len(cat) == 0


def test_filter_catalogue_static():
    """Test filtering of catalogues (static parameters)."""
    # Tag filter
    cat = katpoint.Catalogue(TARGETS)
    cat = cat.filter(tags=["~radec"])
    num_not_radec = len([t for t in TARGETS if "radec" not in t])
    assert len(cat.targets) == num_not_radec, "Number of targets incorrect"
    cat = katpoint.Catalogue(TARGETS)
    cat = cat.filter(tags=["special"])
    num_special = len([t for t in TARGETS if "special" in t])
    assert len(cat.targets) == num_special, "Number of targets incorrect"
    # Flux filter
    cat.add(FLUX_TARGET)
    cat2 = cat.filter(flux_limit=50 * u.Jy, flux_frequency=1.5 * u.MHz)
    assert len(cat2.targets) == 1, "Number of targets with sufficient flux should be 1"
    assert cat != cat2, "Catalogues should be inequal"
    with pytest.raises(ValueError):
        cat.filter(flux_limit=[0, 50, 100] * u.Jy)  # too many limits
    with pytest.raises(ValueError):
        cat.filter(flux_limit=[0, 50] * u.Jy)  # no flux frequency


def test_filter_catalogue_dynamic():
    """Test filtering of catalogues (dynamic parameters)."""
    cat = katpoint.Catalogue(TARGETS).filter(tags="special")
    cat.add(FLUX_TARGET)
    cat3 = cat.filter(az_limit=[0, 180] * u.deg, timestamp=TIMESTAMP, antenna=ANTENNA)
    assert len(cat3.targets) == 1, "Number of targets rising should be 1"
    cat4 = cat.filter(az_limit=[180, 0] * u.deg, timestamp=TIMESTAMP, antenna=ANTENNA)
    assert len(cat4.targets) == 5, "Number of targets setting should be 5"
    with pytest.raises(ValueError):
        cat.filter(az_limit=0 * u.deg)  # too few limits
    with pytest.raises(ValueError):
        cat.filter(az_limit=[0, 90, 180] * u.deg)  # too many limits
    cat.add(katpoint.Target("Zenith, azel, 0, 90"))
    cat5 = cat.filter(el_limit=85 * u.deg, timestamp=TIMESTAMP, antenna=ANTENNA)
    assert len(cat5.targets) == 1, "Number of targets close to zenith should be 1"
    with pytest.raises(ValueError):
        cat.filter(el_limit=[0, 20, 40] * u.deg)  # too many limits
    sun = katpoint.Target("Sun, special")
    cat6 = cat.filter(
        dist_limit=[0.0, 1.0] * u.deg,
        proximity_targets=sun,
        timestamp=TIMESTAMP,
        antenna=ANTENNA,
    )
    assert len(cat6.targets) == 1, "Number of targets close to Sun should be 1"
    with pytest.raises(ValueError):
        cat.filter(dist_limit=[0.0, 1.0, 2.0] * u.deg)  # too many limits
    with pytest.raises(ValueError):
        cat.filter(dist_limit=[0.0, 1.0] * u.deg)  # no proximity targets


def test_sort_catalogue():
    """Test sorting of catalogues."""
    cat = katpoint.Catalogue(TARGETS)
    assert len(cat.targets) == len(TARGETS)
    cat1 = cat.sort(key="name")
    assert cat1 == cat, "Catalogue equality failed"
    assert cat1.targets[0].name == "Acamar", "Sorting on name failed"
    cat2 = cat.sort(key="ra", timestamp=TIMESTAMP, antenna=ANTENNA)
    assert cat2.targets[0].name == "Alpheratz", "Sorting on ra failed"
    cat3 = cat.sort(key="dec", timestamp=TIMESTAMP, antenna=ANTENNA)
    assert cat3.targets[0].name == "Miaplacidus", "Sorting on dec failed"
    cat4 = cat.sort(key="az", timestamp=TIMESTAMP, antenna=ANTENNA, ascending=False)
    assert cat4.targets[0].name == "Polaris", "Sorting on az failed"  # az: 359:25:07.3
    cat5 = cat.sort(key="el", timestamp=TIMESTAMP, antenna=ANTENNA)
    assert cat5.targets[-1].name == "Zenith", "Sorting on el failed"  # el: 90:00:00.0
    cat.add(FLUX_TARGET)
    cat6 = cat.sort(key="flux", ascending=False, flux_frequency=1.5 * u.MHz)
    assert "flux" in (
        cat6.targets[0].name,
        cat6.targets[-1].name,
    ), "Flux target should be at start or end of catalogue after sorting"
    assert (
        cat6.targets[0].flux_density(1.5 * u.MHz) == 100.0 * u.Jy
        or cat6.targets[-1].flux_density(1.5 * u.MHz) == 100.0 * u.Jy
    ), "Sorting on flux failed"


def test_visibility_list():
    """Test output of visibility list."""
    antenna2 = katpoint.Antenna(
        "XDM2, -25:53:23.05075, 27:41:03.36453, 1406.1086, 15.0, 100.0 0.0 0.0"
    )
    cat = katpoint.Catalogue(TARGETS)
    cat.add(FLUX_TARGET)
    cat.remove("Zenith")
    cat.visibility_list(
        timestamp=TIMESTAMP,
        antenna=ANTENNA,
        flux_frequency=1.5 * u.MHz,
        antenna2=antenna2,
    )
    cat.antenna = ANTENNA
    cat.flux_frequency = 1.5 * u.MHz
    cat.visibility_list(timestamp=TIMESTAMP)
    with pytest.raises(TypeError):
        cat.flux_frequency = 1.5
