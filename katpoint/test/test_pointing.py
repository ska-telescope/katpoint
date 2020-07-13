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

"""Tests for the pointing module."""

import numpy as np
import pytest

import katpoint

from .helper import assert_angles_almost_equal


class TestPointingModel:
    """Test pointing model."""

    def setup(self):
        np.random.seed(42)
        az_range = katpoint.deg2rad(np.arange(-185.0, 275.0, 5.0))
        el_range = katpoint.deg2rad(np.arange(0.0, 86.0, 1.0))
        mesh_az, mesh_el = np.meshgrid(az_range, el_range)
        self.az = mesh_az.ravel()
        self.el = mesh_el.ravel()
        # Generate random parameter values with this spread
        self.param_stdev = katpoint.deg2rad(20. / 60.)
        self.num_params = len(katpoint.PointingModel())

    def test_pointing_model_load_save(self):
        """Test construction / load / save of pointing model."""
        params = katpoint.deg2rad(np.random.randn(self.num_params + 1))
        pm = katpoint.PointingModel(params[:-1])
        print('%r %s' % (pm, pm))
        pm2 = katpoint.PointingModel(params[:-2])
        assert pm2.values()[-1] == 0.0, 'Unspecified pointing model params not zeroed'
        pm3 = katpoint.PointingModel(params)
        assert pm3.values()[-1] == params[-2], (
            'Superfluous pointing model params not handled correctly')
        pm4 = katpoint.PointingModel(pm.description)
        assert pm4.description == pm.description, (
            'Saving pointing model to string and loading it again failed')
        assert pm4 == pm, 'Pointing models should be equal'
        assert pm2 != pm, 'Pointing models should be inequal'
        # np.testing.assert_almost_equal(pm4.values(), pm.values(), decimal=6)
        for (v4, v) in zip(pm4.values(), pm.values()):
            if type(v4) == float:
                np.testing.assert_almost_equal(v4, v, decimal=6)
            else:
                np.testing.assert_almost_equal(v4.rad, v, decimal=6)
        try:
            assert hash(pm4) == hash(pm), 'Pointing model hashes not equal'
        except TypeError:
            pytest.fail('PointingModel object not hashable')

    def test_pointing_closure(self):
        """Test closure between pointing correction and its reverse operation."""
        # Generate random pointing model
        params = self.param_stdev * np.random.randn(self.num_params)
        pm = katpoint.PointingModel(params)
        # Test closure on (az, el) grid
        pointed_az, pointed_el = pm.apply(self.az, self.el)
        az, el = pm.reverse(pointed_az, pointed_el)
        assert_angles_almost_equal(az, self.az, decimal=6,
                                   err_msg='Azimuth closure error for params=%s' % (params,))
        assert_angles_almost_equal(el, self.el, decimal=7,
                                   err_msg='Elevation closure error for params=%s' % (params,))

    def test_pointing_fit(self):
        """Test fitting of pointing model."""
        # Generate random pointing model and corresponding offsets on (az, el) grid
        params = self.param_stdev * np.random.randn(self.num_params)
        params[1] = params[9] = 0.0
        pm = katpoint.PointingModel(params.copy())
        delta_az, delta_el = pm.offset(self.az, self.el)
        enabled_params = (np.arange(self.num_params) + 1).tolist()
        # Comment out these removes, thereby testing more code paths in PointingModel
        # enabled_params.remove(2)
        # enabled_params.remove(10)
        fitted_params, sigma_params = pm.fit(self.az, self.el, delta_az, delta_el, enabled_params=[])
        np.testing.assert_equal(fitted_params, np.zeros(self.num_params))
        fitted_params, sigma_params = pm.fit(self.az, self.el, delta_az, delta_el, enabled_params=enabled_params)
        np.testing.assert_almost_equal(fitted_params, params, decimal=9)
