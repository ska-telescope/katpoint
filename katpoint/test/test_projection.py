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

"""Tests for the projection module."""

import numpy as np
from numpy import pi as PI   # Unorthodox but shortens those parametrization lines a lot
import pytest

import katpoint

from .helper import assert_angles_almost_equal

try:
    from .aips_projection import newpos, dircos
    HAS_AIPS = True
except ImportError:
    HAS_AIPS = False


def random_sphere(N, include_poles=False):
    """Generate `N` random points on a 3D sphere in (longitude, latitude) form."""
    az = PI * (2.0 * np.random.rand(N) - 1.0)
    el = PI * (np.random.rand(N) - 0.5)
    if not include_poles:
        # Keep away from poles (leave them as corner cases)
        el *= 0.999
    return az, el


def random_disk(N, radius_warp, max_theta):
    """Generate `N` random points on a 2D circular disk in (x, y) form."""
    theta = max_theta * np.random.rand(N)
    phi = 2 * PI * np.random.rand(N)
    r = radius_warp(theta)
    return r * np.cos(phi), r * np.sin(phi)


def generate_data_sin(N):
    """Generate test data for orthographic (SIN) projection."""
    az0, el0 = random_sphere(N)
    # (x, y) points within unit circle
    x, y = random_disk(N, np.sin, max_theta=PI/2)
    return az0, el0, x, y


def generate_data_tan(N):
    """Generate test data for gnomonic (TAN) projection."""
    az0, el0 = random_sphere(N)
    # Perform inverse TAN mapping to spread out points on plane
    # Stay away from edge of hemisphere
    x, y = random_disk(N, np.tan, max_theta=PI/2 - 0.01)
    return az0, el0, x, y


def generate_data_arc(N):
    """Generate test data for zenithal equidistant (ARC) projection."""
    az0, el0 = random_sphere(N)
    # (x, y) points within circle of radius pi
    # Stay away from edge of circle
    x, y = random_disk(N, lambda theta: theta, max_theta=PI - 0.01)
    return az0, el0, x, y


def generate_data_stg(N):
    """Generate test data for stereographic (STG) projection."""
    az0, el0 = random_sphere(N)
    # Perform inverse STG mapping to spread out points on plane
    # Stay well away from point of projection
    x, y = random_disk(N, lambda theta: 2.0 * np.sin(theta) / (1.0 + np.cos(theta)),
                       max_theta=0.8 * PI)
    return az0, el0, x, y


def generate_data_car(N):
    """Generate test data for plate carree (CAR) projection."""
    # Unrestricted (az0, el0) points on sphere
    az0, el0 = random_sphere(N, include_poles=True)
    # Unrestricted (x, y) points on corresponding plane
    x, y = random_sphere(N, include_poles=True)
    return az0, el0, x, y


def generate_data_ssn(N):
    """Generate test data for swapped orthographic (SSN) projection."""
    az0, el0 = random_sphere(N)
    # (x, y) points within complicated SSN domain - clipped unit circle
    cos_el0 = np.cos(el0)
    # The x coordinate is bounded by +- cos(el0)
    x = (2.0 * np.random.rand(N) - 1.0) * cos_el0
    # The y coordinate ranges between two (semi-)circles centred on origin:
    # the unit circle on one side and circle of radius cos(el0) on other side
    y_offset = -np.sqrt(cos_el0 ** 2 - x ** 2)
    y_range = -y_offset + np.sqrt(1.0 - x ** 2)
    y = (y_range * np.random.rand(N) + y_offset) * np.sign(el0)
    return az0, el0, x, y


generate_data = {'SIN': generate_data_sin, 'TAN': generate_data_tan,
                 'ARC': generate_data_arc, 'STG': generate_data_stg,
                 'CAR': generate_data_car, 'SSN': generate_data_ssn}


# The decimal accuracy for each projection is the maximum that makes
# the test pass during an extended random run.
@pytest.mark.parametrize('projection, decimal',
                         [('SIN', 10), ('TAN', 8), ('ARC', 8),
                          ('STG', 9), ('CAR', 12), ('SSN', 10)])
def test_random_closure(projection, decimal, N=100):
    """Do random projections and check closure."""
    plane_to_sphere = katpoint.plane_to_sphere[projection]
    sphere_to_plane = katpoint.sphere_to_plane[projection]
    az0, el0, x, y = generate_data[projection](N)
    az, el = plane_to_sphere(az0, el0, x, y)
    xx, yy = sphere_to_plane(az0, el0, az, el)
    aa, ee = plane_to_sphere(az0, el0, xx, yy)
    np.testing.assert_almost_equal(x, xx, decimal=decimal)
    np.testing.assert_almost_equal(y, yy, decimal=decimal)
    assert_angles_almost_equal(az, aa, decimal=decimal)
    assert_angles_almost_equal(el, ee, decimal=decimal)


# The decimal accuracy for each projection is the maximum that makes
# the test pass during an extended random run.
@pytest.mark.skipif(not HAS_AIPS, reason="AIPS projection module not found")
@pytest.mark.parametrize('projection, aips_code, decimal',
                         [('SIN', 2, 9), ('TAN', 3, 10), ('ARC', 4, 8), ('STG', 6, 9)])
def test_aips_compatibility(projection, aips_code, decimal, N=100):
    """Compare with original AIPS routine (if available)."""
    plane_to_sphere = katpoint.plane_to_sphere[projection]
    sphere_to_plane = katpoint.sphere_to_plane[projection]
    az0, el0, x, y = generate_data[projection](N)
    if projection == 'TAN':
        # AIPS TAN only deprojects (x, y) coordinates within unit circle
        r = x * x + y * y
        az0, el0 = az0[r <= 1.0], el0[r <= 1.0]
        x, y = x[r <= 1.0], y[r <= 1.0]
    az, el = plane_to_sphere(az0, el0, x, y)
    xx, yy = sphere_to_plane(az0, el0, az, el)
    az_aips, el_aips = np.zeros_like(az), np.zeros_like(el)
    x_aips, y_aips = np.zeros_like(xx), np.zeros_like(yy)
    for n, _ in enumerate(az):
        az_aips[n], el_aips[n], ierr = newpos(aips_code, az0[n], el0[n], x[n], y[n])
        assert ierr == 0
        x_aips[n], y_aips[n], ierr = dircos(aips_code, az0[n], el0[n], az[n], el[n])
        assert ierr == 0
    # AIPS NEWPOS STG has poor accuracy on azimuth angle (large closure errors by itself)
    if projection != 'STG':
        assert_angles_almost_equal(az, az_aips, decimal=decimal)
    assert_angles_almost_equal(el, el_aips, decimal=decimal)
    np.testing.assert_almost_equal(xx, x_aips, decimal=decimal)
    np.testing.assert_almost_equal(yy, y_aips, decimal=decimal)


@pytest.mark.parametrize(
    "projection, sphere, plane",
    [
        # Reference point at pole on sphere
        ('SSN', (0.0, PI/2, 0.0, 0.0), [0.0, 1.0]),
        ('SSN', (0.0, PI/2, PI, 1e-12), [0.0, 1.0]),
        ('SSN', (0.0, PI/2, PI/2, 0.0), [0.0, 1.0]),
        ('SSN', (0.0, PI/2, -PI/2, 0.0), [0.0, 1.0]),
    ]
)
def test_sphere_to_plane(projection, sphere, plane):
    """Test specific cases (sphere -> plane)."""
    sphere_to_plane = katpoint.sphere_to_plane[projection]
    xy = np.array(sphere_to_plane(*sphere))
    np.testing.assert_almost_equal(xy, plane, decimal=12)


@pytest.mark.parametrize("projection, sphere", [('ARC', (0.0, 0.0, 0.0, PI))])
def test_sphere_to_plane_invalid(projection, sphere):
    """Test points outside allowed domain on sphere (sphere -> plane)."""
    sphere_to_plane = katpoint.sphere_to_plane[projection]
    with pytest.raises(ValueError):
        sphere_to_plane(*sphere)


@pytest.mark.parametrize("projection", ['SIN', 'TAN', 'STG', 'SSN'])
def test_sphere_to_plane_outside_domain(projection):
    """Test points outside allowed domain on sphere (sphere -> plane)."""
    test_sphere_to_plane_invalid(projection, (0.0, 0.0, PI, 0.0))
    test_sphere_to_plane_invalid(projection, (0.0, 0.0, 0.0, PI))


def test_sphere_to_plane_special():
    """Test special corner cases (sphere -> plane)."""
    sphere_to_plane = katpoint.sphere_to_plane['ARC']
    # Point diametrically opposite the reference point on sphere
    xy = np.array(sphere_to_plane(PI, 0.0, 0.0, 0.0))
    np.testing.assert_almost_equal(np.abs(xy), [PI, 0.0], decimal=12)


@pytest.mark.parametrize(
    "projection, plane, sphere",
    [
        # Points on circle with radius pi in plane
        ('ARC', (0.0, 0.0, PI, 0.0), [PI, 0.0]),
        ('ARC', (0.0, 0.0, -PI, 0.0), [-PI, 0.0]),
        ('ARC', (0.0, 0.0, 0.0, PI), [PI, 0.0]),
        ('ARC', (0.0, 0.0, 0.0, -PI), [PI, 0.0]),
        # Reference point at pole on sphere
        ('SSN', (0.0, PI/2, 0.0, 1.0), [0.0, 0.0]),
        ('SSN', (0.0, -PI/2, 0.0, -1.0), [0.0, 0.0]),
        # Test valid (x, y) domain
        ('SSN', (0.0, 1.0, 0.0, -np.cos(1.0)), [0.0, PI/2]),
        ('SSN', (0.0, -1.0, 0.0, np.cos(1.0)), [0.0, -PI/2]),
    ]
)
def test_plane_to_sphere(projection, plane, sphere):
    """Test specific cases (plane -> sphere)."""
    plane_to_sphere = katpoint.plane_to_sphere[projection]
    ae = np.array(plane_to_sphere(*plane))
    assert_angles_almost_equal(ae, sphere, decimal=12)


def plane_to_sphere_invalid(projection, plane):
    """Test points outside allowed domain in plane (plane -> sphere)."""
    plane_to_sphere = katpoint.plane_to_sphere[projection]
    with pytest.raises(ValueError):
        plane_to_sphere(*plane)


@pytest.mark.parametrize("projection, offset_p", [('SIN', 2.0), ('ARC', 4.0), ('SSN', 2.0)])
def test_plane_to_sphere_outside_domain(projection, offset_p):
    """Test points outside allowed domain in plane (plane -> sphere)."""
    plane_to_sphere_invalid(projection, (0.0, 0.0, offset_p, 0.0))
    plane_to_sphere_invalid(projection, (0.0, 0.0, 0.0, offset_p))


def sphere_to_plane_to_sphere(projection, reference, sphere, plane):
    """Project from sphere to plane and back again and check results on both legs."""
    test_sphere_to_plane(projection, tuple(reference) + tuple(sphere), plane)
    test_plane_to_sphere(projection, tuple(reference) + tuple(plane), sphere)


@pytest.mark.parametrize("projection, offset_s, offset_p",
                         [('SIN', PI/2, 1.0), ('TAN', PI/4, 1.0), ('ARC', PI/2, PI/2),
                          ('STG', PI/2, 2.0), ('SSN', PI/2, -1.0)])
def test_sphere_to_plane_to_sphere_origin(projection, offset_s, offset_p):
    """Test five-point cross along axes, centred on origin (sphere -> plane -> sphere)."""
    sphere_to_plane_to_sphere(projection, (0.0, 0.0), (0.0, 0.0), [0.0, 0.0])
    sphere_to_plane_to_sphere(projection, (0.0, 0.0), (+offset_s, 0.0), [+offset_p, 0.0])
    sphere_to_plane_to_sphere(projection, (0.0, 0.0), (-offset_s, 0.0), [-offset_p, 0.0])
    sphere_to_plane_to_sphere(projection, (0.0, 0.0), (0.0, +offset_s), [0.0, +offset_p])
    sphere_to_plane_to_sphere(projection, (0.0, 0.0), (0.0, -offset_s), [0.0, -offset_p])


@pytest.mark.parametrize("projection, offset_s, offset_p",
                         [('SIN', PI/2 - 1e-12, 1.0), ('TAN', PI/4, 1.0),
                          ('ARC', PI/2, PI/2), ('STG', PI/2, 2.0)])
def test_sphere_to_plane_to_sphere_pole(projection, offset_s, offset_p):
    """Test four-point cross along axes, centred on pole (sphere -> plane -> sphere)."""
    el = PI/2 - offset_s
    sphere_to_plane_to_sphere(projection, (0.0, PI/2), (+PI/2, el), [+offset_p, 0.0])
    sphere_to_plane_to_sphere(projection, (0.0, PI/2), (-PI/2, el), [-offset_p, 0.0])
    sphere_to_plane_to_sphere(projection, (0.0, PI/2), (PI, el), [0.0, +offset_p])
    sphere_to_plane_to_sphere(projection, (0.0, PI/2), (0.0, el), [0.0, -offset_p])


def sphere_to_plane_original_ssn(target_az, target_el, scan_az, scan_el):
    """Mattieu's original version of SSN projection."""
    ll = np.cos(target_el) * np.sin(target_az - scan_az)
    mm = np.cos(target_el) * np.sin(scan_el) * np.cos(
        target_az - scan_az) - np.cos(scan_el) * np.sin(target_el)
    return ll, mm


def plane_to_sphere_original_ssn(target_az, target_el, ll, mm):
    """Mattieu's original version of SSN projection."""
    scan_az = target_az - np.arcsin(np.clip(ll / np.cos(target_el), -1.0, 1.0))
    scan_el = np.arcsin(np.clip(
        (np.sqrt(1.0 - ll**2 - mm**2) * np.sin(target_el)
         + np.sqrt(np.cos(target_el)**2 - ll**2) * mm) / (1.0 - ll**2), -1.0, 1.0))
    return scan_az, scan_el


def test_vs_original_ssn(decimal=10, N=100):
    """SSN projection: compare against Mattieu's original version."""
    plane_to_sphere = katpoint.plane_to_sphere['SSN']
    az0, el0, x, y = generate_data['SSN'](N)
    az, el = plane_to_sphere(az0, el0, x, y)
    ll, mm = sphere_to_plane_original_ssn(az0, el0, az, el)
    aa, ee = plane_to_sphere_original_ssn(az0, el0, ll, mm)
    np.testing.assert_almost_equal(x, ll, decimal=decimal)
    np.testing.assert_almost_equal(y, -mm, decimal=decimal)
    assert_angles_almost_equal(az, aa, decimal=decimal)
    assert_angles_almost_equal(el, ee, decimal=decimal)
