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

"""Shared pytest utilities."""

import numpy as np
from astropy.coordinates import UnitSphericalRepresentation


def assert_angles_almost_equal(x, y, **kwargs):
    """Check that two angles / arrays are almost equal (modulo 2 pi)."""
    def primary_angle(x):
        return x - np.round(x / (2.0 * np.pi)) * 2.0 * np.pi
    np.testing.assert_almost_equal(primary_angle(x - y), np.zeros(np.shape(x)), **kwargs)


def check_separation(actual, lon, lat, tol):
    """Check that actual and desired directions are within tolerance."""
    desired = actual.realize_frame(UnitSphericalRepresentation(lon, lat))
    assert actual.separation(desired) <= tol
