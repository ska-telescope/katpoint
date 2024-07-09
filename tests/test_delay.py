################################################################################
# Copyright (c) 2014,2017-2024, National Research Foundation (SARAO)
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

"""Tests for the delay modules."""

import json
from io import StringIO

import astropy.units as u
import numpy as np
import pytest
from astropy import __version__ as astropy_version
from astropy.coordinates import Angle
from packaging.version import Version

import katpoint

try:
    from almacalc.highlevel import calc
except ImportError:
    HAS_ALMACALC = False
else:
    HAS_ALMACALC = True


def test_construct_save_load():
    """Test construction / save / load of delay model."""
    m = katpoint.DelayModel("1.0, -2.0, -3.0, 4.123, 5.0, 6.0")
    m.header["date"] = "2014-01-15"
    # An empty file should lead to a BadModelFile exception
    cfg_file = StringIO()
    with pytest.raises(katpoint.BadModelFile):
        m.fromfile(cfg_file)
    m.tofile(cfg_file)
    cfg_str = cfg_file.getvalue()
    cfg_file.close()
    # Load the saved config file
    cfg_file = StringIO(cfg_str)
    m2 = katpoint.DelayModel()
    m2.fromfile(cfg_file)
    assert m == m2, "Saving delay model to file and loading it again failed"
    params = m.delay_params
    m3 = katpoint.DelayModel()
    m3.fromdelays(params)
    assert (
        m == m3
    ), "Converting delay model to delay parameters and loading it again failed"
    try:
        assert hash(m) == hash(m3), "Delay model hashes not equal"
    except TypeError:
        pytest.fail("DelayModel object not hashable")


# Special stationary target nominally perpendicular to A2-A1 baseline.
# This also has the least diurnal aberration:
# - az = -90 deg => 1e-18 s
# - az = 90 deg => 5e-18 s
# - az = 0 deg => 4e-14 s ...
TARGET1 = katpoint.Target.from_azel("-90:00:00.0", "00:00:00.0")
TARGET2 = katpoint.Target("Sun, special")
ANT1 = katpoint.Antenna("A1, -31.0, 18.0, 0.0, 12.0, 0.0 0.0 0.0")
ANT2 = katpoint.Antenna("A2, -31.0, 18.0, 0.0, 12.0, 0.0 10.0 0.0")
ANT3 = katpoint.Antenna("A3, -31.0, 18.0, 0.0, 12.0, 5.0 10.0 3.0")
TS = katpoint.Timestamp("2013-08-14 08:25")
DELAYS = katpoint.DelayCorrection([ANT2, ANT3], ANT1, 1.285 * u.GHz)


def test_construction():
    """Test construction of DelayCorrection object."""
    descr = DELAYS.description
    assert DELAYS.inputs == ["A2h", "A2v", "A3h", "A3v"]
    assert DELAYS.ant_locations.shape == (2,), "Ant_locations property has wrong size"
    assert DELAYS.ant_locations[0] == ANT2.location, "Wrong location for first antenna"
    assert DELAYS.ant_locations[1] == ANT3.location, "Wrong location for second antenna"
    assert DELAYS.ref_location == ANT1.location, "Wrong reference location"
    delays2 = katpoint.DelayCorrection(descr)
    delays_dict = json.loads(descr)
    delays2_dict = json.loads(delays2.description)
    assert (
        delays2_dict == delays_dict
    ), "Objects created through description strings differ"
    with pytest.raises(ValueError):
        katpoint.DelayCorrection("")
    delays3 = katpoint.DelayCorrection([], ANT1)
    assert delays3.ant_locations.shape == (0,), "Ant_locations property has wrong size"
    d = delays3.delays(TARGET1, TS + np.arange(3))
    assert d.shape == (0, 3), "Delay correction with no antennas should fail gracefully"
    # Check construction with different antenna reference positions
    delays4 = katpoint.DelayCorrection([ANT1, ANT2], ANT3)
    ant1_vs_ant3 = np.array(delays4.ant_models["A1"].values())
    ant3_vs_ant1 = np.array(DELAYS.ant_models["A3"].values())
    assert np.allclose(ant3_vs_ant1, -ant1_vs_ant3, rtol=0, atol=2e-5)
    delays5 = katpoint.DelayCorrection([ANT1, ANT2])
    assert delays5.ref_ant == ANT1
    # Check that older extra_delay attribute still works in description dict
    older_dict = dict(delays_dict)
    older_dict["extra_delay"] = older_dict["extra_correction"]
    del older_dict["extra_correction"]
    delays6 = katpoint.DelayCorrection(json.dumps(older_dict))
    assert delays6.extra_correction == DELAYS.extra_correction
    assert DELAYS.tropospheric_model == "None"
    del older_dict["extra_delay"]
    with pytest.raises(KeyError):
        katpoint.DelayCorrection(json.dumps(older_dict))


def test_delays():
    """Test delay calculations."""
    delay0 = DELAYS.delays(TARGET1, TS)
    assert delay0.shape == (4,)
    assert np.allclose(delay0[:2], 0.0, rtol=0, atol=2e-18)
    delay1 = DELAYS.delays(TARGET1, [TS - 1.0, TS, TS + 1.0])
    assert delay1.shape == (4, 3)
    assert np.allclose(delay1[:2, :], 0.0, rtol=0, atol=1e-18)
    delay_now = DELAYS.delays(TARGET1, None)
    assert np.allclose(delay_now[:2], delay0[:2], rtol=0, atol=3e-18)
    assert np.allclose(delay_now[2:], delay0[2:], rtol=2e-10, atol=0)


def test_correction():
    """Test delay correction."""
    extra_correction = DELAYS.extra_correction
    delay0, phase0, drate0, frate0 = DELAYS.corrections(TARGET1, TS)
    delay1, phase1, drate1, frate1 = DELAYS.corrections(TARGET1, [TS, TS + 1.0])
    # First check dimensions for time dimension T0 = () and T1 = (2,), respectively
    assert np.shape(delay0["A2h"]) == np.shape(phase0["A2h"]) == ()
    assert np.shape(drate0["A2h"]) == np.shape(frate0["A2h"]) == (0,)
    assert np.shape(delay1["A2h"]) == np.shape(phase1["A2h"]) == (2,)
    assert np.shape(drate1["A2h"]) == np.shape(frate1["A2h"]) == (1,)
    # This target is special - direction basically perpendicular to baseline
    # (and stationary)
    assert np.allclose(delay0["A2h"], extra_correction, rtol=0, atol=1e-6 * u.ps)
    assert np.allclose(delay0["A2v"], extra_correction, rtol=0, atol=1e-6 * u.ps)
    assert np.allclose(drate1["A2h"], [0.0], rtol=0, atol=1e-22)
    assert np.allclose(drate1["A2v"], [0.0], rtol=0, atol=1e-22)
    assert np.allclose(frate1["A2h"], [0.0], rtol=0, atol=1e-12 * u.rad / u.s)
    assert np.allclose(frate1["A2v"], [0.0], rtol=0, atol=1e-12 * u.rad / u.s)
    assert np.allclose(
        delay1["A2h"], extra_correction.repeat(2), rtol=0, atol=1e-6 * u.ps
    )
    assert np.allclose(
        delay1["A2v"], extra_correction.repeat(2), rtol=0, atol=1e-6 * u.ps
    )
    # Compare to target geometric delay calculations
    delay0, _, _, _ = DELAYS.corrections(TARGET2, TS)
    _, _, drate1, _ = DELAYS.corrections(TARGET2, (TS - 0.5, TS + 0.5))
    tgt_delay, tgt_delay_rate = TARGET2.geometric_delay(ANT2, TS, ANT1)
    assert np.allclose(
        delay0["A2h"], extra_correction - tgt_delay, rtol=0, atol=0.03 * u.ps
    )
    assert np.allclose(drate1["A2h"][0], -tgt_delay_rate, rtol=0, atol=1e-18)


def test_offset():
    """Test target offset."""
    assert np.allclose(
        DELAYS.delays(TARGET1, TS, offset=None),
        DELAYS.delays(TARGET1, TS, offset={}),
        atol=0,
        rtol=1e-15,
    )
    azel = TARGET1.azel(TS, ANT1)
    offset = dict(projection_type="SIN")
    target3 = katpoint.Target.from_azel(
        azel.az - Angle(1.0, unit=u.deg), azel.alt - Angle(1.0, unit=u.deg)
    )
    x, y = target3.sphere_to_plane(azel.az.rad, azel.alt.rad, TS, ANT1, **offset)
    offset["x"] = x
    offset["y"] = y
    extra_correction = DELAYS.extra_correction
    delay0, _, _, _ = DELAYS.corrections(target3, TS, offset=offset)
    delay1, _, drate1, _ = DELAYS.corrections(target3, (TS, TS + 1.0), offset)
    # Conspire to return to special target1
    assert np.allclose(delay0["A2h"], extra_correction, rtol=0, atol=1e-6 * u.ps)
    assert np.allclose(delay0["A2v"], extra_correction, rtol=0, atol=1e-6 * u.ps)
    assert np.allclose(
        delay1["A2h"], extra_correction.repeat(2), rtol=0, atol=1e-6 * u.ps
    )
    assert np.allclose(
        delay1["A2v"], extra_correction.repeat(2), rtol=0, atol=1e-6 * u.ps
    )
    assert np.allclose(drate1["A2h"], [0.0], rtol=0, atol=1e-22)
    assert np.allclose(drate1["A2v"], [0.0], rtol=0, atol=1e-22)
    # Now try (ra, dec) coordinate system
    radec = TARGET1.radec(TS, ANT1)
    offset = dict(projection_type="ARC", coord_system="radec")
    target4 = katpoint.Target.from_radec(
        radec.ra - Angle(1.0, unit=u.deg), radec.dec - Angle(1.0, unit=u.deg)
    )
    x, y = target4.sphere_to_plane(radec.ra.rad, radec.dec.rad, TS, ANT1, **offset)
    offset["x"] = x
    offset["y"] = y
    extra_correction = DELAYS.extra_correction
    delay0, _, _, _ = DELAYS.corrections(target4, TS, offset=offset)
    delay1, _, drate1, _ = DELAYS.corrections(target4, (TS, TS + 1.0), offset)
    # Conspire to return to special target1
    assert np.allclose(delay0["A2h"], extra_correction, rtol=0, atol=1e-6 * u.ps)
    assert np.allclose(delay0["A2v"], extra_correction, rtol=0, atol=1e-6 * u.ps)
    assert np.allclose(delay1["A2h"][0], extra_correction, rtol=0, atol=1e-6 * u.ps)
    assert np.allclose(delay1["A2v"][0], extra_correction, rtol=0, atol=1e-6 * u.ps)
    assert np.allclose(drate1["A2h"], [0.0], rtol=0, atol=5e-12)
    assert np.allclose(drate1["A2v"], [0.0], rtol=0, atol=5e-12)


# Target and antennas used in katpoint_vs_calc study
TARGET = katpoint.Target("J1939-6342, radec, 19:39:25.03, -63:42:45.6")
DELAY_MODEL = dict(
    ref_ant="array, -30:42:39.8, 21:26:38, 1086.6, 0",
    extra_correction=0.0,
    sky_centre_freq=1284000000.0,
    tropospheric_model="SaastamoinenZenithDelay-GlobalMappingFunction",
)
ANT_MODELS = dict(
    m048="-2805.653 2686.863 -9.7545 0 0 1.2",
    m058="2805.764 2686.873 -3.6595 0 0 1",
    s0105="8539.425 -1430.292 -5.872 0 0 2",  # had 39 ps error in 1.0a1
    s0121="-3545.28803 -10207.44399 -9.18584 0 0 2",
)
WEATHER = dict(temperature=20 * u.deg_C, pressure=1000 * u.mbar, relative_humidity=0.5)


def test_tropospheric_delay():
    """Test that `DelayCorrection` correctly applies tropospheric delays."""
    model = dict(ant_models=ANT_MODELS, **DELAY_MODEL)
    dc = katpoint.DelayCorrection(json.dumps(model))
    tropospheric_delay = katpoint.troposphere.delay.TroposphericDelay(dc.ref_location)
    elevation = 15 * u.deg
    target = katpoint.Target.from_azel(0, elevation.to_value(u.rad))
    ts = katpoint.Timestamp(1605646800.0).time
    expected_delay = tropospheric_delay(elevation=elevation, timestamp=ts, **WEATHER)
    delay = dc.delays(target, ts, **WEATHER) - dc.delays(target, ts)
    assert np.allclose(delay, expected_delay, rtol=0, atol=1e-8 * u.ps)
    # The combination of positive relative humidity and
    # unspecified temperature is an error.
    no_temperature = WEATHER.copy()
    del no_temperature["temperature"]
    with pytest.raises(ValueError):
        dc.delays(target, ts, **no_temperature)
    with pytest.raises(ValueError):
        dc.corrections(target, ts, **no_temperature)


# XXX Check Solar System / TLE bodies too by going via radec for Calc
@pytest.mark.skipif(not HAS_ALMACALC, reason="almacalc is not installed")
@pytest.mark.parametrize(
    "times,ant_models,geom_atol,tropo_atol",
    [
        (
            1605680300.0 + np.linspace(0, 50000, 6),  # minimum elevation is 15 degrees
            {"m063": "-3419.5 -1840.4 16.3 0 0 1"},
            0.4 * u.ps,
            0.4 * u.ps,
        ),
        # Check tropospheric delays at 5 degrees elevation. The main difference is the
        # TroposphericDelay location which is m063 for Calc and refant for katpoint.
        (1605668400.0, {"m063": "-3419.5 -1840.4 16.3"}, 0.15 * u.ps, 5 * u.ps),
        # Let the two antennas be the same, and the tropospheric results are much closer
        (1605668400.0, {"ref": ""}, 1e-8 * u.ps, 0.4 * u.ps),
        # Use antennas and times from the katpoint_vs_calc study
        (1571219913.0 + np.arange(0, 54000, 6000), ANT_MODELS, 1.2 * u.ps, 0.1 * u.ps),
    ],
)
def test_against_calc(times, ant_models, geom_atol, tropo_atol):
    """Test geocentric katpoint against Calc (especially tropos and NIAO parts)."""
    times = katpoint.Timestamp(times)
    if ant_models != {"ref": ""}:
        pytest.xfail("Topocentric delay models don't match Calc")
    # Check the basic geometric contribution, without NIAO or troposphere
    model_enu = dict(
        ant_models={k: " ".join(v.split()[:3]) for k, v in ant_models.items()},
        **DELAY_MODEL
    )
    model_enu["tropospheric_model"] = "None"
    dc = katpoint.DelayCorrection(json.dumps(model_enu))
    geom_delay = dc.delays(TARGET, times)[::2]
    expected_geom_delay = calc(
        dc.ant_locations, TARGET.body.coord, times.time, dc.ref_location
    ).T
    assert np.allclose(geom_delay, expected_geom_delay, rtol=0, atol=geom_atol)

    # Check the NIAO contribution independent of
    # geometric and tropospheric contributions.
    model = dict(ant_models=ant_models, **DELAY_MODEL)
    model["tropospheric_model"] = None
    dc = katpoint.DelayCorrection(json.dumps(model))
    # Vector of axis offsets per antenna
    niao = np.array([dm["NIAO"] for dm in dc.ant_models.values()]) * u.m
    delay = dc.delays(TARGET, times)[::2]
    expected_delay = calc(
        dc.ant_locations,
        TARGET.body.coord,
        times.time,
        dc.ref_location,
        axis_offset=niao,
    ).T
    niao_delay = delay - geom_delay
    expected_niao_delay = expected_delay - expected_geom_delay
    assert np.allclose(niao_delay, expected_niao_delay, rtol=0, atol=0.005 * u.ps)

    # Check the tropospheric contribution independent of geometric or NIAO contributions
    model_tropo = model_enu.copy()
    model_tropo["tropospheric_model"] = DELAY_MODEL["tropospheric_model"]
    dc = katpoint.DelayCorrection(json.dumps(model_tropo))
    delay = dc.delays(TARGET, times, **WEATHER)[::2]
    expected_delay = calc(
        dc.ant_locations, TARGET.body.coord, times.time, dc.ref_location, **WEATHER
    ).T
    tropo_delay = delay - geom_delay
    expected_tropo_delay = expected_delay - expected_geom_delay
    assert np.allclose(tropo_delay, expected_tropo_delay, rtol=0, atol=tropo_atol)


TLE_TARGET = (
    "GPS BIIA-21 (PRN 09), tle, "
    "1 22700U 93042A   07266.32333151  .00000012  00000-0  10000-3 0  8054, "
    "2 22700  55.4408  61.3790 0191986  78.1802 283.9935  2.00561720104282"
)
astropy_version = Version(astropy_version)


@pytest.mark.parametrize(
    "description",
    ["azel, 10, -10", "radec, 20, -20", "gal, 30, -30", "Moon, special", TLE_TARGET],
)
def test_astropy_broadcasting(description):
    """Check that various Bodies can handle multiple times and antennas."""
    times = katpoint.Timestamp(1605646800.0 + np.linspace(0, 86400, 4).reshape(1, 4))
    ant_models = {
        "m048": "-2805.653 2686.863 -9.7545",
        "m058": "2805.764 2686.873 -3.6595",
        "s0121": "-3545.28803 -10207.44399 -9.18584",
    }
    model = dict(ant_models=ant_models, **DELAY_MODEL)
    dc = katpoint.DelayCorrection(json.dumps(model))
    target = katpoint.Target(description)
    expected_shape = (2 * len(ant_models),) + times.time.shape
    delay = dc.delays(target, times)
    assert delay.shape == expected_shape
    # Do a basic null offset check to verify additional coordinate transformation paths
    offset = dict(x=0.0, y=0.0, projection_type="TAN", coord_system="radec")
    # XXX Astropy < 4.3 has AltAz errors on nearby objects (see astropy/astropy#10994).
    # Since delays are based on (az, el), `delay` is actually out by 20 ps on the Moon
    # and 80 ps on the GPS satellite, and the radec offset delay is correct...
    tol = 0.0001 * u.ps if astropy_version >= Version("4.3") else 100 * u.ps
    assert np.allclose(dc.delays(target, times, offset), delay, rtol=0, atol=tol)
    offset = dict(x=0.0, y=0.0, projection_type="STG", coord_system="azel")
    assert np.allclose(
        dc.delays(target, times, offset), delay, rtol=0, atol=0.0001 * u.ps
    )
    pressure = np.zeros_like(times.time) * u.hPa
    temperature = np.ones_like(times.time) * 20 * u.deg_C
    relative_humidity = np.zeros_like(times.time) * u.dimensionless_unscaled
    assert np.allclose(
        dc.delays(target, times, None, pressure, temperature, relative_humidity),
        delay,
        rtol=0,
        atol=0.0001 * u.ps,
    )
