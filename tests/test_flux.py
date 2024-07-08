################################################################################
# Copyright (c) 2013,2017-2021,2023, National Research Foundation (SARAO)
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

"""Tests for the flux module."""

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

import katpoint

DESCRIPTION = "(1.0 2.0 2.0 0.0 0.0 0.0 0.0 0.0 2.0 0.5 0.25 -0.75)"
FLUX_MODEL = katpoint.FluxDensityModel.from_description(DESCRIPTION)
FLUX_TARGET = katpoint.Target("radec, 0.0, 0.0, " + FLUX_MODEL.description)
NO_FLUX_TARGET = katpoint.Target("radec, 0.0, 0.0")


def test_construct():
    """Test valid and invalid flux model constructions."""
    assert FLUX_MODEL.description == DESCRIPTION
    with pytest.raises(TypeError):
        katpoint.FluxDensityModel(1.0, 2.0, [2.0])
    with pytest.raises(katpoint.FluxError):
        katpoint.FluxDensityModel.from_description("a b c")


def test_unit_model():
    """Test unit flux model, as well as comparisons and hashes."""
    unit_model = katpoint.FluxDensityModel(100 * u.MHz, 200 * u.MHz, [0.0])
    unit_model2 = katpoint.FluxDensityModel(100 * u.MHz, 200 * u.MHz, [])
    assert unit_model.flux_density(110 * u.MHz) == 1.0 * u.Jy, "Flux calculation wrong"
    # At least one coefficient is always shown
    assert unit_model.description == "(100.0 200.0 0.0)"
    assert unit_model == unit_model2, "Flux models not equal"
    try:
        assert hash(unit_model) == hash(unit_model2), "Flux model hashes not equal"
    except TypeError:
        pytest.fail("FluxDensityModel object not hashable")


def test_too_many_params():
    """Test flux model with too many parameters."""
    with pytest.warns(FutureWarning):
        too_many_params = katpoint.FluxDensityModel.from_description(
            "(1.0 2.0 2.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0)"
        )
    # Must truncate default coefficients, including I=1
    assert too_many_params.description == "(1.0 2.0 2.0)"
    assert (
        too_many_params.flux_density(1.5 * u.MHz) == 100.0 * u.Jy
    ), "Flux calculation for too many params wrong"


def test_too_few_params():
    """Test flux model with too few parameters."""
    with pytest.raises(katpoint.FluxError):
        katpoint.FluxDensityModel.from_description("(1.0)")
    too_few_params = katpoint.FluxDensityModel.from_description("(1.0 2.0 2.0)")
    assert (
        too_few_params.flux_density(1.5 * u.MHz) == 100.0 * u.Jy
    ), "Flux calculation for too few params wrong"


def test_flux_density():
    """Test flux density calculation."""
    assert (
        FLUX_MODEL.flux_density(1.5 * u.MHz) == 200.0 * u.Jy
    ), "Flux calculation wrong"
    np.testing.assert_equal(
        FLUX_MODEL.flux_density([1.5, 1.5] * u.MHz),
        np.array([200.0, 200.0]) * u.Jy,
        "Flux calculation for multiple frequencies wrong",
    )
    np.testing.assert_equal(
        FLUX_MODEL.flux_density([0.5, 2.5] * u.MHz),
        np.array([np.nan, np.nan]) * u.Jy,
        "Flux calculation for out-of-range frequencies wrong",
    )
    with pytest.raises(ValueError):
        NO_FLUX_TARGET.flux_density()
    np.testing.assert_equal(
        NO_FLUX_TARGET.flux_density([1.5, 1.5] * u.MHz),
        np.array([np.nan, np.nan]) * u.Jy,
        "Empty flux model leads to wrong empty flux shape",
    )
    FLUX_TARGET.flux_frequency = 1.5 * u.MHz
    assert (
        FLUX_TARGET.flux_density() == 200.0 * u.Jy
    ), "Flux calculation for default freq wrong"
    with pytest.raises(TypeError):
        FLUX_TARGET.flux_frequency = 1.5
    print(FLUX_TARGET)


def test_flux_density_stokes():
    """Test flux density calculation for Stokes parameters."""
    np.testing.assert_array_equal(
        FLUX_MODEL.flux_density_stokes(1.5 * u.MHz),
        np.array([200.0, 50.0, 25.0, -75.0]) * u.Jy,
    )
    np.testing.assert_array_equal(
        FLUX_MODEL.flux_density_stokes([1.0, 1.5, 3.0] * u.MHz),
        np.array(
            [
                [200.0, 50.0, 25.0, -75.0],
                [200.0, 50.0, 25.0, -75.0],
                [np.nan, np.nan, np.nan, np.nan],
            ]
        )
        * u.Jy,
    )
    with pytest.raises(ValueError):
        NO_FLUX_TARGET.flux_density_stokes()
    np.testing.assert_array_equal(
        NO_FLUX_TARGET.flux_density_stokes(1.5 * u.MHz),
        np.array([np.nan, np.nan, np.nan, np.nan]) * u.Jy,
        "Empty flux model leads to wrong empty flux shape",
    )
    np.testing.assert_array_equal(
        NO_FLUX_TARGET.flux_density_stokes([1.5, 1.5] * u.MHz),
        np.array([[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]])
        * u.Jy,
        "Empty flux model leads to wrong empty flux shape",
    )
    FLUX_TARGET.flux_frequency = 1.5 * u.MHz
    np.testing.assert_array_equal(
        FLUX_TARGET.flux_density_stokes(),
        np.array([200.0, 50.0, 25.0, -75.0]) * u.Jy,
        "Flux calculation for default freq wrong",
    )


def test_wavelength_inputs():
    """Check that flux densities can be calculated for wavelengths."""
    model = katpoint.FluxDensityModel(100 * u.MHz, 200 * u.MHz, [0.0, 1.0])
    freq = 150 * u.MHz
    assert np.allclose(model.flux_density(freq), 150 * u.Jy, rtol=1e-15)
    wavelength = const.c / freq
    assert np.allclose(model.flux_density(wavelength), 150 * u.Jy, rtol=1e-15)
    model2 = katpoint.FluxDensityModel(
        const.c / (100 * u.MHz), const.c / (200 * u.MHz), [0.0, 1.0]
    )
    freq = 150 * u.MHz
    assert np.allclose(model2.flux_density(freq), 150 * u.Jy, rtol=1e-15)
    wavelength = const.c / freq
    assert np.allclose(model2.flux_density(wavelength), 150 * u.Jy, rtol=1e-15)
