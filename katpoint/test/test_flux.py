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


class TestFluxDensityModel:
    """Test flux density model calculation."""

    def setup(self):
        self.unit_model = katpoint.FluxDensityModel(100., 200., [0.])
        self.unit_model2 = katpoint.FluxDensityModel(100., 200., [0.])
        self.flux_model = katpoint.FluxDensityModel(
            '(1.0 2.0 2.0 0.0 0.0 0.0 0.0 0.0 2.0 0.5 0.25 -0.75)')
        with pytest.warns(FutureWarning):
            self.too_many_params = katpoint.FluxDensityModel(
                '(1.0 2.0 2.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0)')
        self.too_few_params = katpoint.FluxDensityModel('(1.0 2.0 2.0)')
        self.flux_target = katpoint.Target('radec, 0.0, 0.0, ' + self.flux_model.description)
        self.no_flux_target = katpoint.Target('radec, 0.0, 0.0')

    def test_construct(self):
        with pytest.raises(ValueError):
            katpoint.FluxDensityModel('1.0 2.0 2.0', 2.0, [2.0])
        with pytest.raises(ValueError):
            katpoint.FluxDensityModel('1.0')

    def test_description(self):
        assert self.flux_model.description == '(1.0 2.0 2.0 0.0 0.0 0.0 0.0 0.0 2.0 0.5 0.25 -0.75)'
        # Must truncate default coefficients, including I=1
        assert self.too_many_params.description == '(1.0 2.0 2.0)'
        # At least one coefficient is always shown
        assert self.unit_model.description == '(100.0 200.0 0.0)'

    def test_flux_density(self):
        """Test flux density calculation."""
        assert self.unit_model.flux_density(110.) == 1.0, 'Flux calculation wrong'
        assert self.flux_model.flux_density(1.5) == 200.0, 'Flux calculation wrong'
        assert self.too_many_params.flux_density(1.5) == 100.0, (
            'Flux calculation for too many params wrong')
        assert self.too_few_params.flux_density(1.5) == 100.0, (
            'Flux calculation for too few params wrong')
        np.testing.assert_equal(self.flux_model.flux_density([1.5, 1.5]),
                                np.array([200.0, 200.0]),
                                'Flux calculation for multiple frequencies wrong')
        np.testing.assert_equal(self.flux_model.flux_density([0.5, 2.5]),
                                np.array([np.nan, np.nan]),
                                'Flux calculation for out-of-range frequencies wrong')
        with pytest.raises(ValueError):
            self.no_flux_target.flux_density()
        np.testing.assert_equal(self.no_flux_target.flux_density([1.5, 1.5]),
                                np.array([np.nan, np.nan]),
                                'Empty flux model leads to wrong empty flux shape')
        self.flux_target.flux_freq_MHz = 1.5
        assert self.flux_target.flux_density() == 200.0, 'Flux calculation for default freq wrong'
        print(self.flux_target)

    def test_flux_density_stokes(self):
        """Test flux density calculation for Stokes parameters"""
        np.testing.assert_array_equal(self.flux_model.flux_density_stokes(1.5),
                                      np.array([200.0, 50.0, 25.0, -75.0]))
        np.testing.assert_array_equal(self.flux_model.flux_density_stokes([1.0, 1.5, 3.0]),
                                      np.array([[200.0, 50.0, 25.0, -75.0],
                                                [200.0, 50.0, 25.0, -75.0],
                                                [np.nan, np.nan, np.nan, np.nan]]))
        with pytest.raises(ValueError):
            self.no_flux_target.flux_density_stokes()
        np.testing.assert_array_equal(self.no_flux_target.flux_density_stokes(1.5),
                                      np.array([np.nan, np.nan, np.nan, np.nan]),
                                      'Empty flux model leads to wrong empty flux shape')
        np.testing.assert_array_equal(self.no_flux_target.flux_density_stokes([1.5, 1.5]),
                                      np.array([[np.nan, np.nan, np.nan, np.nan],
                                                [np.nan, np.nan, np.nan, np.nan]]),
                                      'Empty flux model leads to wrong empty flux shape')
        self.flux_target.flux_freq_MHz = 1.5
        np.testing.assert_array_equal(self.flux_target.flux_density_stokes(),
                                      np.array([200.0, 50.0, 25.0, -75.0]),
                                      'Flux calculation for default freq wrong')

    def test_compare(self):
        assert self.unit_model == self.unit_model2, 'Flux models not equal'

    def test_hash(self):
        try:
            assert hash(self.unit_model) == hash(self.unit_model2), 'Flux model hashes not equal'
        except TypeError:
            pytest.fail('FluxDensityModel object not hashable')
