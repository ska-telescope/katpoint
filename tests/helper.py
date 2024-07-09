################################################################################
# Copyright (c) 2020-2021,2023, National Research Foundation (SARAO)
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

import astropy.units as u
import numpy as np
from astropy.coordinates import UnitSphericalRepresentation


def assert_angles_almost_equal(x, y, **kwargs):
    """Check that two angles / arrays are almost equal (modulo 2 pi)."""

    def primary_angle(x):
        return x - np.round(x / (2.0 * np.pi)) * 2.0 * np.pi

    x = np.asarray(x)
    y = np.asarray(y)
    np.testing.assert_array_equal(
        0 * x, 0 * y, "Array shapes and/or NaN patterns differ"
    )
    d = primary_angle(np.nan_to_num(x - y))
    np.testing.assert_almost_equal(d, np.zeros(np.shape(x)), **kwargs)


def _lon_lat_str(coord):
    """Print coordinates in the format they are specified in unit tests."""
    lookup = {v: k for k, v in coord.representation_component_names.items()}
    lon = getattr(coord, lookup["lon"])
    lat = getattr(coord, lookup["lat"])
    kwargs = dict(sep=":", pad=True)
    if lookup["lon"] == "ra":
        lon_str = lon.to_string(unit=u.hourangle, precision=5, **kwargs) + "h"
    else:
        lon_str = lon.to_string(unit=u.deg, precision=4, **kwargs) + "d"
    lat_str = lat.to_string(unit=u.deg, precision=4, **kwargs) + "d"
    return f"({lon_str}, {lat_str})"


def check_separation(actual, lon, lat, tol):
    """Check that actual and desired directions are within tolerance."""
    desired = actual.realize_frame(UnitSphericalRepresentation(lon, lat))
    sep = actual.separation(desired)
    assert sep <= tol, (
        f"Expected {_lon_lat_str(desired)}, got {_lon_lat_str(actual)} "
        f"which is {sep.to(tol.unit):.3f} away instead of within {tol}"
    )
