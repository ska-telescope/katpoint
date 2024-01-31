################################################################################
# Copyright (c) 2009-2022, National Research Foundation (SARAO)
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

"""Tests for the refraction module."""

from dataclasses import FrozenInstanceError

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import ICRS, AltAz, Angle, EarthLocation

import katpoint
from katpoint.troposphere.delay import (
    GlobalMappingFunction,
    SaastamoinenZenithDelay,
    TroposphericDelay,
)
from katpoint.troposphere.refraction import HaystackRefraction

try:
    from almacalc.highlevel import calc
    from almacalc.lowlevel import gmf11, sastd, sastw
except ImportError:
    HAS_ALMACALC = False
else:
    HAS_ALMACALC = True

from .helper import assert_angles_almost_equal


def test_refraction_basic():
    """Test basic refraction correction properties."""
    tropo = katpoint.TroposphericRefraction()
    print(repr(tropo))
    with pytest.raises(ValueError):
        katpoint.TroposphericRefraction("unknown")
    tropo2 = katpoint.TroposphericRefraction()
    assert tropo == tropo2, "Refraction models should be equal"
    try:
        assert hash(tropo) == hash(tropo2), "Refraction model hashes should be equal"
    except TypeError:
        pytest.fail("TroposphericRefraction object not hashable")
    with pytest.raises(FrozenInstanceError):
        tropo.model_id = "it's frozen, so the model_id can't be changed"


@pytest.mark.parametrize(
    "el,pressure,temp,humidity,refracted_el",
    [
        # Produced by katpoint 0.10
        (15.0 * u.deg, 900 * u.hPa, 10 * u.deg_C, 0.80, 15.061621833721594 * u.deg),
        (15.0 * u.deg, 900 * u.hPa, 20 * u.deg_C, 0.25, 15.056928675685779 * u.deg),
        (15.0 * u.deg, 900 * u.hPa, 40 * u.deg_C, 0.10, 15.060931376413768 * u.deg),
    ],
)
def test_refraction_haystack(el, pressure, temp, humidity, refracted_el):
    """Check for any regressions in Haystack refraction model."""
    measured_el = HaystackRefraction.refract(el, pressure, temp, humidity)
    np.testing.assert_array_almost_equal_nulp(measured_el, refracted_el)


def test_refraction_closure():
    """Test closure between refraction correction and its reverse operation."""
    tropo = katpoint.TroposphericRefraction()
    el = Angle(np.arange(0.0, 90.1, 0.1), u.deg)
    azel = AltAz(az=Angle(np.zeros_like(el)), alt=el)
    # Generate random meteorological data (a single measurement, hopefully sensible)
    pressure = (900.0 + 200.0 * np.random.rand()) * u.hPa
    temp = (-10.0 + 50.0 * np.random.rand()) * u.deg_C
    humidity = (5.0 + 90.0 * np.random.rand()) * u.percent
    # Test closure on el grid
    refracted_azel = tropo.refract(azel, pressure, temp, humidity)
    reversed_azel = tropo.unrefract(refracted_azel, pressure, temp, humidity)
    np.testing.assert_allclose(
        reversed_azel.alt,
        el,
        rtol=0.0,
        atol=0.03 * u.arcsec,
        err_msg="Elevation closure error for "
        f"pressure={pressure:g}, temp={temp:g}, humidity={humidity:g}",
    )
    # Generate random meteorological data,
    # now one weather measurement per elevation value.
    pressure = (900.0 + 200.0 * np.random.rand(len(el))) * u.hPa
    temp = (-10.0 + 50.0 * np.random.rand(len(el))) * u.deg_C
    humidity = (5.0 + 90.0 * np.random.rand(len(el))) * u.percent
    # Test closure on el grid
    refracted_azel = tropo.refract(azel, pressure, temp, humidity)
    reversed_azel = tropo.unrefract(refracted_azel, pressure, temp, humidity)
    np.testing.assert_allclose(
        reversed_azel.alt,
        el,
        rtol=0.0,
        atol=0.03 * u.arcsec,
        err_msg="Elevation closure error for "
        f"pressure={pressure}, temp={temp}, humidity={humidity}",
    )


def test_delay_basic():
    """Test basic tropospheric delay properties."""
    with pytest.raises(TypeError):
        TroposphericDelay()  # pylint: disable=no-value-for-parameter
    location = EarthLocation.from_geodetic("-30:42:39.8", "21:26:38.0", "1086.6")
    with pytest.raises(ValueError):
        TroposphericDelay(location, "bad_format")
    with pytest.raises(ValueError):
        TroposphericDelay(location, "unknown-components")
    tropo = TroposphericDelay(location)
    print(repr(tropo))
    location2 = EarthLocation.from_geodetic("-30:42:39.8", "21:26:38.0", "1086.6")
    tropo2 = TroposphericDelay(location2)
    assert tropo == tropo2, "Tropospheric delay models should be equal by value"
    # TroposphericDelay is not hashable yet (prevented by EarthLocation)
    with pytest.raises(TypeError):
        hash(tropo)
    with pytest.raises(FrozenInstanceError):
        tropo.model_id = "it's frozen, so the model_id can't be changed"


_locations_and_times = [
    ("-25:53:23.0", "27:41:03.0", 1406.0, "2020-12-25"),
    ("35:00:00.0", "-40:00:00.0", 0.0, "2019-01-28"),
    ("85:00:00.0", "170:00:00.0", -200.0, "2000-06-01"),
    ("-35:00:00.0", "-40:00:00.0", 6000.0, "2001-03-14"),
    ("-90:00:00.0", "180:00:00.0", 200.0, 1234567890.0),
]


@pytest.mark.skipif(not HAS_ALMACALC, reason="almacalc is not installed")
@pytest.mark.parametrize("latitude,longitude,height,timestamp", _locations_and_times)
def test_zenith_delay(
    latitude, longitude, height, timestamp
):  # pylint: disable=unused-argument
    """Test hydrostatic and wet zenith delays against AlmaCalc."""
    location = EarthLocation.from_geodetic(longitude, latitude, height)
    zd = SaastamoinenZenithDelay(location)
    pressure = np.arange(800.0, 1000.0, 5.0) * u.hPa
    actual = zd.hydrostatic(pressure)
    expected = (
        sastd(pressure.value, location.lat.rad, location.height.value) * u.m / const.c
    )
    assert np.allclose(actual, expected, rtol=0, atol=0.01 * u.ps)
    # Check alternative units
    pressure = np.arange(0.8, 1.0, 0.005) * u.bar
    actual = zd.hydrostatic(pressure)
    assert np.allclose(actual, expected, rtol=0, atol=0.01 * u.ps)
    temperature, relative_humidity = np.meshgrid(
        np.arange(-5.0, 45.0, 5.0) * u.deg_C, np.arange(0.0, 1.05, 0.05)
    )
    actual = zd.wet(temperature, relative_humidity)
    expected = sastw(relative_humidity, temperature.value) * u.m / const.c
    assert np.allclose(actual, expected, rtol=0, atol=0.15 * u.ps)
    # Add a little realism to check the practical impact of tweaks to wet zenith delay
    dry_site = relative_humidity <= 2.06 - temperature / (20 * u.deg_C)
    assert np.allclose(actual[dry_site], expected[dry_site], rtol=0, atol=0.05 * u.ps)
    # Check alternative units
    temperature, relative_humidity = np.meshgrid(
        (np.arange(-5.0, 45.0, 5.0) + 273.15) * u.K,
        np.arange(0.0, 105.0, 5.0) * u.percent,
    )
    actual = zd.wet(temperature, relative_humidity)
    assert np.allclose(actual, expected, rtol=0, atol=0.15 * u.ps)


@pytest.mark.skipif(not HAS_ALMACALC, reason="almacalc is not installed")
@pytest.mark.parametrize("latitude,longitude,height,timestamp", _locations_and_times)
def test_mapping_function(latitude, longitude, height, timestamp):
    """Test hydrostatic and wet mapping functions against AlmaCalc."""
    location = EarthLocation.from_geodetic(longitude, latitude, height)
    elevation = np.arange(1, 91) * u.deg
    mf = GlobalMappingFunction(location)
    actual_hydrostatic = mf.hydrostatic(elevation, timestamp)
    actual_wet = mf.wet(elevation, timestamp)
    time = katpoint.Timestamp(timestamp).time
    expected_hydrostatic, expected_wet = gmf11(
        time.utc.jd,
        location.lat.rad,
        location.lon.rad,
        location.height.to_value(u.m),
        elevation.to_value(u.rad),
    )
    # Measure relative tolerance because mapping function is a scale factor
    assert np.allclose(actual_hydrostatic, expected_hydrostatic, rtol=1e-9)
    assert np.allclose(actual_wet, expected_wet, rtol=1e-8)


_default_model = "SaastamoinenZenithDelay-GlobalMappingFunction"


@pytest.mark.skipif(not HAS_ALMACALC, reason="almacalc is not installed")
@pytest.mark.parametrize(
    "model_id,elevation,atol",
    [
        # Calc spits out NaNs at 90 degrees elevation (probably due to arcsin in ATMG)
        (_default_model + "-hydrostatic", 89.99 * u.deg, 0.01 * u.ps),
        (_default_model + "-hydrostatic", 30 * u.deg, 0.01 * u.ps),
        (_default_model + "-hydrostatic", 15 * u.deg, 0.01 * u.ps),
        # The wet comparison suffers at low elevations
        # due to slight tweaks to Python version.
        (_default_model + "-wet", 89.99 * u.deg, 0.02 * u.ps),
        (_default_model + "-wet", 30 * u.deg, 0.03 * u.ps),
        (_default_model + "-wet", 15 * u.deg, 0.05 * u.ps),
        (_default_model + "-total", 89.99 * u.deg, 0.02 * u.ps),
        (_default_model, 65 * u.deg, 0.02 * u.ps),
        (_default_model, 45 * u.deg, 0.02 * u.ps),
        (_default_model, 30 * u.deg, 0.03 * u.ps),
        (_default_model, 15 * u.deg, 0.06 * u.ps),
        (_default_model, 5 * u.deg, 0.25 * u.ps),
        (_default_model, 1 * u.deg, 1.2 * u.ps),
    ],
)
def test_tropospheric_delay(model_id, elevation, atol):
    """Test hydrostatic and wet tropospheric delays against AlmaCalc."""
    location = EarthLocation.from_geodetic("21:26:38.0", "-30:42:39.8", 1086.6)
    temperature = 25.0 * u.deg_C
    pressure = 905.0 * u.hPa
    relative_humidity = 0.2
    timestamp = katpoint.Timestamp("2020-11-01 22:13:00")
    obstime = timestamp.time
    td = TroposphericDelay(location, model_id)
    actual = td(pressure, temperature, relative_humidity, elevation, timestamp)
    azel = AltAz(az=0 * u.deg, alt=elevation, location=location, obstime=obstime)
    radec = azel.transform_to(ICRS())
    enable_dry_delay = not model_id.endswith("-wet")
    enable_wet_delay = not model_id.endswith("-hydrostatic")
    expected = calc(
        location,
        radec,
        obstime,
        temperature=temperature,
        pressure=enable_dry_delay * pressure,
        relative_humidity=enable_wet_delay * relative_humidity,
    )
    assert np.allclose(actual, expected, rtol=0, atol=atol)
