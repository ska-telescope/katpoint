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

"""Tests for the flux module."""

import numpy as np
import pytest

import katpoint


FLUX_MODEL = katpoint.FluxDensityModel('(1.0 2.0 2.0 0.0 0.0 0.0 0.0 0.0 2.0 0.5 0.25 -0.75)')
FLUX_TARGET = katpoint.Target('radec, 0.0, 0.0, ' + FLUX_MODEL.description)
NO_FLUX_TARGET = katpoint.Target('radec, 0.0, 0.0')


def test_construct():
    """Test valid and invalid flux model constructions."""
    assert FLUX_MODEL.description == '(1.0 2.0 2.0 0.0 0.0 0.0 0.0 0.0 2.0 0.5 0.25 -0.75)'
    with pytest.raises(ValueError):
        katpoint.FluxDensityModel('1.0 2.0 2.0', 2.0, [2.0])
    with pytest.raises(ValueError):
        katpoint.FluxDensityModel('1.0')


def test_unit_model():
    """Test unit flux model, as well as comparisons and hashes."""
    unit_model = katpoint.FluxDensityModel(100., 200., [0.])
    unit_model2 = katpoint.FluxDensityModel(100., 200., [0.])
    assert unit_model.flux_density(110.) == 1.0, 'Flux calculation wrong'
    # At least one coefficient is always shown
    assert unit_model.description == '(100.0 200.0 0.0)'
    assert unit_model == unit_model2, 'Flux models not equal'
    try:
        assert hash(unit_model) == hash(unit_model2), 'Flux model hashes not equal'
    except TypeError:
        pytest.fail('FluxDensityModel object not hashable')


def test_too_many_params():
    """Test flux model with too many parameters."""
    with pytest.warns(FutureWarning):
        too_many_params = katpoint.FluxDensityModel(
            '(1.0 2.0 2.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0)')
    # Must truncate default coefficients, including I=1
    assert too_many_params.description == '(1.0 2.0 2.0)'
    assert too_many_params.flux_density(1.5) == 100.0, 'Flux calculation for too many params wrong'


def test_too_few_params():
    """Test flux model with too few parameters."""
    too_few_params = katpoint.FluxDensityModel('(1.0 2.0 2.0)')
    assert too_few_params.flux_density(1.5) == 100.0, 'Flux calculation for too few params wrong'


def test_flux_density():
    """Test flux density calculation."""
    assert FLUX_MODEL.flux_density(1.5) == 200.0, 'Flux calculation wrong'
    np.testing.assert_equal(FLUX_MODEL.flux_density([1.5, 1.5]), np.array([200.0, 200.0]),
                            'Flux calculation for multiple frequencies wrong')
    np.testing.assert_equal(FLUX_MODEL.flux_density([0.5, 2.5]), np.array([np.nan, np.nan]),
                            'Flux calculation for out-of-range frequencies wrong')
    with pytest.raises(ValueError):
        NO_FLUX_TARGET.flux_density()
    np.testing.assert_equal(NO_FLUX_TARGET.flux_density([1.5, 1.5]), np.array([np.nan, np.nan]),
                            'Empty flux model leads to wrong empty flux shape')
    FLUX_TARGET.flux_freq_MHz = 1.5
    assert FLUX_TARGET.flux_density() == 200.0, 'Flux calculation for default freq wrong'
    print(FLUX_TARGET)


def test_flux_density_stokes():
    """Test flux density calculation for Stokes parameters"""
    np.testing.assert_array_equal(FLUX_MODEL.flux_density_stokes(1.5),
                                  np.array([200.0, 50.0, 25.0, -75.0]))
    np.testing.assert_array_equal(FLUX_MODEL.flux_density_stokes([1.0, 1.5, 3.0]),
                                  np.array([[200.0, 50.0, 25.0, -75.0],
                                            [200.0, 50.0, 25.0, -75.0],
                                            [np.nan, np.nan, np.nan, np.nan]]))
    with pytest.raises(ValueError):
        NO_FLUX_TARGET.flux_density_stokes()
    np.testing.assert_array_equal(NO_FLUX_TARGET.flux_density_stokes(1.5),
                                  np.array([np.nan, np.nan, np.nan, np.nan]),
                                  'Empty flux model leads to wrong empty flux shape')
    np.testing.assert_array_equal(NO_FLUX_TARGET.flux_density_stokes([1.5, 1.5]),
                                  np.array([[np.nan, np.nan, np.nan, np.nan],
                                            [np.nan, np.nan, np.nan, np.nan]]),
                                  'Empty flux model leads to wrong empty flux shape')
    FLUX_TARGET.flux_freq_MHz = 1.5
    np.testing.assert_array_equal(FLUX_TARGET.flux_density_stokes(),
                                  np.array([200.0, 50.0, 25.0, -75.0]),
                                  'Flux calculation for default freq wrong')
