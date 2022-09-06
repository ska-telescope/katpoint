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

"""Tests for the pointing module."""

import pytest
import numpy as np

import katpoint

from .helper import assert_angles_almost_equal


@pytest.fixture(name='pointing_grid')
def fixture_pointing_grid():
    """Generate a grid of (az, el) values in natural antenna coordinates."""
    az_range = np.radians(np.arange(-185.0, 275.0, 5.0))
    el_range = np.radians(np.arange(0.0, 86.0, 1.0))
    mesh_az, mesh_el = np.meshgrid(az_range, el_range)
    az = mesh_az.ravel()
    el = mesh_el.ravel()
    return az, el


@pytest.fixture(name='params')
def fixture_params():
    """Generate random parameters for a pointing model."""
    # Generate random parameter values with this spread
    param_stdev = np.radians(20. / 60.)
    num_params = len(katpoint.PointingModel())
    params = param_stdev * np.random.randn(num_params)
    return params


def test_pointing_model_load_save(params):
    """Test construction / load / save of pointing model."""
    pm = katpoint.PointingModel(params)
    print(f'{pm!r} {pm}')
    pm2 = katpoint.PointingModel(params[:-1])
    assert pm2.values()[-1] == 0.0, 'Unspecified pointing model params not zeroed'
    pm3 = katpoint.PointingModel(np.r_[params, 1.0])
    assert pm3.values()[-1] == params[-1], (
        'Superfluous pointing model params not handled correctly')
    pm4 = katpoint.PointingModel(pm.description)
    assert pm4.description == pm.description, (
        'Saving pointing model to string and loading it again failed')
    assert pm4 == pm, 'Pointing models should be equal'
    assert pm2 != pm, 'Pointing models should be inequal'
    # np.testing.assert_almost_equal(pm4.values(), pm.values(), decimal=6)
    for (v4, v) in zip(pm4.values(), pm.values()):
        if isinstance(v4, float):
            np.testing.assert_almost_equal(v4, v, decimal=6)
        else:
            np.testing.assert_almost_equal(v4.rad, v, decimal=6)
    try:
        assert hash(pm4) == hash(pm), 'Pointing model hashes not equal'
    except TypeError:
        pytest.fail('PointingModel object not hashable')


def test_pointing_closure(params, pointing_grid):
    """Test closure between pointing correction and its reverse operation."""
    pm = katpoint.PointingModel(params)
    # Test closure on (az, el) grid
    grid_az, grid_el = pointing_grid
    pointed_az, pointed_el = pm.apply(grid_az, grid_el)
    az, el = pm.reverse(pointed_az, pointed_el)
    assert_angles_almost_equal(az, grid_az, decimal=6,
                               err_msg=f'Azimuth closure error for params={params}')
    assert_angles_almost_equal(el, grid_el, decimal=7,
                               err_msg=f'Elevation closure error for params={params}')


def test_pointing_fit(params, pointing_grid):
    """Test fitting of pointing model."""
    # Generate random pointing model and corresponding offsets on (az, el) grid
    params[1] = params[9] = 0.0
    pm = katpoint.PointingModel(params.copy())
    grid_az, grid_el = pointing_grid
    delta_az, delta_el = pm.offset(grid_az, grid_el)
    enabled_params = (np.arange(len(pm)) + 1).tolist()
    # Comment out these removes, thereby testing more code paths in PointingModel
    # enabled_params.remove(2)
    # enabled_params.remove(10)
    fitted_params, _ = pm.fit(grid_az, grid_el, delta_az, delta_el, enabled_params=[])
    np.testing.assert_equal(fitted_params, np.zeros(len(pm)))
    fitted_params, _ = pm.fit(grid_az, grid_el, delta_az, delta_el, enabled_params=enabled_params)
    np.testing.assert_almost_equal(fitted_params, params, decimal=9)
