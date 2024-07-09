################################################################################
# Copyright (c) 2009-2013,2017-2018,2020-2024, National Research Foundation (SARAO)
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
from astropy.coordinates import CIRS, ICRS, AltAz, Angle, EarthLocation

import katpoint
from katpoint.troposphere.delay import (
    GlobalMappingFunction,
    SaastamoinenZenithDelay,
    TroposphericDelay,
)
from katpoint.troposphere.refraction import (
    _WAVELENGTH_UM,
    ErfaRefraction,
    HaystackRefraction,
)

try:
    from almacalc.highlevel import calc
    from almacalc.lowlevel import gmf11, sastd, sastw
except ImportError:
    HAS_ALMACALC = False
else:
    HAS_ALMACALC = True


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
def test_haystack_refraction(el, pressure, temp, humidity, refracted_el):
    """Check for any regressions in Haystack refraction model."""
    measured_el = HaystackRefraction.refract(
        el.to_value(u.rad), pressure.value, temp.value, humidity
    )
    np.testing.assert_array_almost_equal_nulp(measured_el * u.rad, refracted_el)


@pytest.mark.parametrize(
    "el,pressure,temp,humidity,refracted_el",
    [
        # Produced by katpoint 0.10
        (15.0 * u.deg, 900 * u.hPa, 10 * u.deg_C, 0.80, 15.061429045176592 * u.deg),
        (15.0 * u.deg, 900 * u.hPa, 20 * u.deg_C, 0.25, 15.05540653214649 * u.deg),
        (15.0 * u.deg, 900 * u.hPa, 40 * u.deg_C, 0.10, 15.053121703447712 * u.deg),
    ],
)
def test_erfa_refraction(el, pressure, temp, humidity, refracted_el):
    """Check for any regressions in ERFA refraction model."""
    measured_el = ErfaRefraction.refract(
        el.to_value(u.rad), pressure.value, temp.value, humidity
    )
    np.testing.assert_array_almost_equal_nulp(measured_el * u.rad, refracted_el)


def test_erfa_refraction_against_astropy():
    """Check that our ERFA-derived refraction routines match the Astropy machinery."""
    # Start with a east-pointing dish slewing from near horizon to zenith, in vacuum
    location = EarthLocation.from_geodetic("-30:42:39.8", "21:26:38.0", "1086.6")
    el = Angle(np.arange(5.0, 91.0, 1.0), u.deg)
    az = Angle(np.full_like(el, 90 * u.deg))
    # We have to generate multiple obstimes too if we want multiple weather samples
    # later on, because the refa and refb coefficients are shaped according to obstime.
    # Pick a range of time steps that roughly correspond to celestial tracking.
    time_steps = np.arange(len(el)) * 5.0 * u.min
    obstime = (katpoint.Timestamp("2024-02-01 21:15:00") + time_steps).time
    azel = AltAz(az=az, alt=el, location=location, obstime=obstime)
    # Use CIRS as a reference point, from where we will transform back to Earth.
    # XXX Eventually we could just use topocentric ITRS transforms as a shortcut.
    intermediate = azel.transform_to(CIRS(location=location, obstime=obstime))
    # Obtain vacuum azel from the reference coordinates. It only differs by a tiny
    # amount from `azel`` but this is the direct input to ERFA refraction in Astropy.
    vacuum_azel = intermediate.transform_to(azel)
    # Generate random meteorological data (also try to check different units)
    pressure = (90.0 + 20.0 * np.random.rand(len(vacuum_azel))) * u.kPa
    temp = (-10.0 + 50.0 * np.random.rand(len(vacuum_azel))) * u.deg_C
    humidity = (5.0 + 90.0 * np.random.rand(len(vacuum_azel))) * u.percent
    # Generate a frame that includes location, obstime *and* the weather
    azel_with_weather = azel.replicate_without_data(
        copy=True,
        pressure=pressure,
        temperature=temp,
        relative_humidity=humidity,
        obswl=_WAVELENGTH_UM * u.micron,
    )
    # Obtain refracted azel from the reference coordinates via Astropy
    surface_azel = intermediate.transform_to(azel_with_weather)
    # Apply the same refraction code to the same data via katpoint
    tropo = katpoint.TroposphericRefraction("ErfaRefraction")
    refracted_azel = tropo.refract(vacuum_azel, pressure, temp, humidity)
    # Strip off the weather from frame so that it can be directly compared
    # to the katpoint version without ungoing refraction again in the transform.
    surface_azel = azel.realize_frame(surface_azel.data)
    sep = refracted_azel.separation(surface_azel)
    np.testing.assert_allclose(sep, np.zeros_like(sep), rtol=0.0, atol=1 * u.narcsec)
    # Now pretend that the dish slewed in observed coordinates and find new reference
    surface_azel2 = azel_with_weather.realize_frame(azel.data)
    intermediate2 = surface_azel2.transform_to(CIRS(location=location, obstime=obstime))
    # Come back to vacuum azel, which should be the result of unrefracting surface_azel2
    vacuum_azel2 = intermediate2.transform_to(azel)
    # Unrefract surface_azel2 but azel is easier (same values but no weather to strip)
    unrefracted_azel = tropo.unrefract(azel, pressure, temp, humidity)
    sep2 = unrefracted_azel.separation(vacuum_azel2)
    np.testing.assert_allclose(sep2, np.zeros_like(sep2), rtol=0.0, atol=1 * u.narcsec)


@pytest.mark.parametrize(
    "model,min_elevation,atol",
    [
        ("HaystackRefraction", 0 * u.deg, 0.03 * u.arcsec),
        # ("ErfaRefraction", 5 * u.deg, 0.05 * u.arcsec),  # the advertised consistency
        ("ErfaRefraction", 11 * u.deg, 0.1 * u.arcsec),  # what we found instead
    ],
)
def test_refraction_closure(model, min_elevation, atol):
    """Test closure between refraction correction and its reverse operation."""
    tropo = katpoint.TroposphericRefraction(model)
    el = Angle(np.arange(0.0, 90.1, 0.1), u.deg)
    el = el[el >= min_elevation]
    azel = AltAz(az=Angle(np.zeros_like(el)), alt=el)
    # Generate random meteorological data (a single measurement, hopefully sensible)
    pressure = (900.0 + 200.0 * np.random.rand()) * u.hPa
    temp = (-10.0 + 50.0 * np.random.rand()) * u.deg_C
    humidity = (5.0 + 90.0 * np.random.rand()) * u.percent
    # Test closure on el grid
    refracted_azel = tropo.refract(azel, pressure, temp, humidity)
    reversed_azel = tropo.unrefract(refracted_azel, pressure, temp, humidity)
    msg = (
        "Elevation closure / consistency error for "
        f"pressure={pressure:g}, temp={temp:g}, humidity={humidity:g}"
    )
    np.testing.assert_allclose(reversed_azel.alt, el, rtol=0.0, atol=atol, err_msg=msg)
    # Generate random meteorological data, one weather sample per elevation value.
    # Also test out different units for the parameters.
    pressure = (90.0 + 20.0 * np.random.rand(len(el))) * u.kPa
    temp = (273.15 - 10.0 + 50.0 * np.random.rand(len(el))) * u.K
    humidity = (5.0 + 90.0 * np.random.rand(len(el))) * u.percent
    # Test closure on el grid
    refracted_azel = tropo.refract(azel, pressure, temp, humidity)
    reversed_azel = tropo.unrefract(refracted_azel, pressure, temp, humidity)
    msg = (
        "Elevation closure / consistency error for "
        f"pressure={pressure}, temp={temp}, humidity={humidity}"
    )
    np.testing.assert_allclose(reversed_azel.alt, el, rtol=0.0, atol=atol, err_msg=msg)


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
