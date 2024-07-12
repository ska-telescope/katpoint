################################################################################
# Copyright (c) 2009-2011,2013-2018,2020-2024, National Research Foundation (SARAO)
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

import threading

import numpy as np
import pytest
from astropy.wcs import WCS
from numpy import pi as PI  # Unorthodox but shortens those parametrization lines a lot

import katpoint
from katpoint.projection import (
    OutOfRangeError,
    get_out_of_range_treatment,
    out_of_range_context,
    set_out_of_range_treatment,
    treat_out_of_range_values,
)

from .helper import assert_angles_almost_equal


@pytest.fixture(name="restore_treatment")
def fixture_restore_treatment():
    """Backup existing OutOfRange treatment and restore afterwards."""
    old_treatment = get_out_of_range_treatment()
    yield old_treatment
    set_out_of_range_treatment(old_treatment)


def test_treatment_setup(restore_treatment):
    """Check that we can set out-of-range treatment appropriately."""
    set_out_of_range_treatment("raise")
    assert get_out_of_range_treatment() == "raise"
    set_out_of_range_treatment("nan")
    assert get_out_of_range_treatment() == "nan"
    set_out_of_range_treatment("clip")
    assert get_out_of_range_treatment() == "clip"
    with pytest.raises(ValueError):
        set_out_of_range_treatment("bad treatment")
    with out_of_range_context("raise"):
        assert get_out_of_range_treatment() == "raise"
    assert get_out_of_range_treatment() == "clip"


def test_out_of_range_handling_scalar(restore_treatment):
    """Test out-of-range handling for a scalar value."""
    x = 2
    y = treat_out_of_range_values(x, "Should not happen", lower=0, upper=5)
    np.testing.assert_array_equal(y, x)
    with out_of_range_context("raise"):
        with pytest.raises(OutOfRangeError):
            y = treat_out_of_range_values(x, "Out of range", lower=2.1)
    with out_of_range_context("nan"):
        y = treat_out_of_range_values(x, "Out of range", lower=2.1)
        np.testing.assert_array_equal(y, np.nan)
    with out_of_range_context("clip"):
        y = treat_out_of_range_values(x, "Out of range", upper=1.1)
        np.testing.assert_array_equal(y, 1.1)


def test_out_of_range_handling_array(restore_treatment):
    """Test out-of-range handling for an array of values."""
    x = [1, 2, 3, 4]
    y = treat_out_of_range_values(x, "Should not happen", lower=0, upper=5)
    np.testing.assert_array_equal(y, x)
    with out_of_range_context("raise"):
        with pytest.raises(OutOfRangeError):
            y = treat_out_of_range_values(x, "Out of range", lower=2.1)
    with out_of_range_context("nan"):
        y = treat_out_of_range_values(x, "Out of range", lower=2.1)
        np.testing.assert_array_equal(y, [np.nan, np.nan, 3.0, 4.0])
    with out_of_range_context("clip"):
        y = treat_out_of_range_values(x, "Out of range", upper=1.1)
        np.testing.assert_array_equal(y, [1.0, 1.1, 1.1, 1.1])


@pytest.mark.parametrize("x, scalar", [(2.0, True), (np.array(2.0), False)])
def test_scalar_vs_0d(x, scalar, restore_treatment):
    """Check that out-of-range treatment preserves scalarity."""
    with out_of_range_context("clip"):
        y = treat_out_of_range_values(x, "Out of range", upper=1.1)
        assert np.isscalar(y) is scalar


@pytest.mark.parametrize("treatment", ["raise", "nan", "clip"])
def test_clipping_of_minor_outliers(treatment, restore_treatment):
    """Check that out-of-range treatment tolerates extremely small transgressions."""
    x = 1.0 + np.finfo(float).eps
    with out_of_range_context(treatment):
        y = treat_out_of_range_values(x, "Should not trigger false alarm", upper=1.0)
        assert y == 1.0


def test_out_of_range_initialisation_in_new_thread():
    """Check out-of-range treatment behaviour with multiple threads."""

    def my_thread():
        try:
            result.append(treat_out_of_range_values(2.0, "Should raise", upper=1.0))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            result.append(exc)

    result = []
    thread = threading.Thread(target=my_thread)
    with out_of_range_context("nan"):
        # Make sure the thread code runs inside our out_of_range_context
        thread.start()
        thread.join()
    assert isinstance(result[0], OutOfRangeError)


def random_sphere(random, N, include_poles=False):
    """Generate `N` random points on a 3D sphere in (longitude, latitude) form."""
    az = PI * (2.0 * random.rand(N) - 1.0)
    el = PI * (random.rand(N) - 0.5)
    if not include_poles:
        # Keep away from poles (leave them as corner cases)
        el *= 0.999
    return az, el


def random_disk(random, N, radius_warp, max_theta):
    """Generate `N` random points on a 2D circular disk in (x, y) form."""
    theta = max_theta * random.rand(N)
    phi = 2 * PI * random.rand(N)
    r = radius_warp(theta)
    return r * np.cos(phi), r * np.sin(phi)


def generate_data_sin(random, N):
    """Generate test data for orthographic (SIN) projection."""
    az0, el0 = random_sphere(random, N)
    # (x, y) points within unit circle
    x, y = random_disk(random, N, np.sin, max_theta=PI / 2)
    return az0, el0, x, y


def generate_data_tan(random, N):
    """Generate test data for gnomonic (TAN) projection."""
    az0, el0 = random_sphere(random, N)
    # Perform inverse TAN mapping to spread out points on plane
    # Stay away from edge of hemisphere
    x, y = random_disk(random, N, np.tan, max_theta=PI / 2 - 0.01)
    return az0, el0, x, y


def generate_data_arc(random, N):
    """Generate test data for zenithal equidistant (ARC) projection."""
    az0, el0 = random_sphere(random, N)
    # (x, y) points within circle of radius pi
    # Stay away from edge of circle
    x, y = random_disk(random, N, lambda theta: theta, max_theta=PI - 0.01)
    return az0, el0, x, y


def generate_data_stg(random, N):
    """Generate test data for stereographic (STG) projection."""
    az0, el0 = random_sphere(random, N)
    # Perform inverse STG mapping to spread out points on plane
    # Stay well away from point of projection
    x, y = random_disk(
        random,
        N,
        lambda theta: 2.0 * np.sin(theta) / (1.0 + np.cos(theta)),
        max_theta=0.8 * PI,
    )
    return az0, el0, x, y


def generate_data_car(random, N):
    """Generate test data for plate carree (CAR) projection."""
    # Unrestricted (az0, el0) points on sphere
    az0, el0 = random_sphere(random, N, include_poles=True)
    # Unrestricted (x, y) points on corresponding plane
    x, y = random_sphere(random, N, include_poles=True)
    return az0, el0, x, y


def generate_data_ssn(random, N):
    """Generate test data for swapped orthographic (SSN) projection."""
    az0, el0 = random_sphere(random, N)
    # (x, y) points within complicated SSN domain - clipped unit circle
    cos_el0 = np.cos(el0)
    # The x coordinate is bounded by +- cos(el0)
    x = (2.0 * random.rand(N) - 1.0) * cos_el0
    # The y coordinate ranges between two (semi-)circles centred on origin:
    # the unit circle on one side and circle of radius cos(el0) on other side
    y_offset = -np.sqrt(cos_el0**2 - x**2)
    y_range = -y_offset + np.sqrt(1.0 - x**2)
    y = (y_range * random.rand(N) + y_offset) * np.sign(el0)
    return az0, el0, x, y


generate_data = {
    "SIN": generate_data_sin,
    "TAN": generate_data_tan,
    "ARC": generate_data_arc,
    "STG": generate_data_stg,
    "CAR": generate_data_car,
    "SSN": generate_data_ssn,
}


# The decimal accuracy for each projection is the maximum that makes
# the test pass during an extended random run (N=100000).
@pytest.mark.parametrize(
    "projection, decimal",
    [("SIN", 10), ("TAN", 10), ("ARC", 10), ("STG", 11), ("CAR", 14), ("SSN", 11)],
)
def test_random_closure(random, projection, decimal, N=100):
    """Do random projections and check closure."""
    plane_to_sphere = katpoint.plane_to_sphere[projection]
    sphere_to_plane = katpoint.sphere_to_plane[projection]
    az0, el0, x, y = generate_data[projection](random, N)
    az, el = plane_to_sphere(az0, el0, x, y)
    xx, yy = sphere_to_plane(az0, el0, az, el)
    aa, ee = plane_to_sphere(az0, el0, xx, yy)
    np.testing.assert_almost_equal(x, xx, decimal=decimal)
    np.testing.assert_almost_equal(y, yy, decimal=decimal)
    assert_angles_almost_equal(az, aa, decimal=decimal)
    assert_angles_almost_equal(el, ee, decimal=decimal)


def _prepare_wcs(code):
    """Construct :class:`~astropy.wcs.WCS` object with `code` projection type."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [0.0, 0.0]
    # Setting the pixel width to one degree ensures that fractional pixel locations
    # match the intermediate world coordinates we actually want.
    wcs.wcs.cdelt = [1.0, 1.0]
    wcs.wcs.ctype = [f"RA---{code}", f"DEC--{code}"]
    return wcs


def _fix_scalar(x, reference):
    """Ensure that `x` is a proper scalar if `reference` is scalar."""
    return x.item() if np.isscalar(reference) else x


def sphere_to_plane_wcs(code, az0_rad, el0_rad, az_rad, el_rad):
    """Project sphere to plane using specified WCSLIB projection.

    Parameters
    ----------
    code : str
        Three-letter projection code defined by the FITS standard
    az0_rad : float or array
        Azimuth / right ascension / longitude of reference point(s), in radians
    el0_rad : float or array
        Elevation / declination / latitude of reference point(s), in radians
    az_rad : float or array
        Azimuth / right ascension / longitude of point(s) to project, in radians
    el_rad : float or array
        Elevation / declination / latitude of point(s) to project, in radians

    Returns
    -------
    x_rad : float or array
        Azimuth-like coordinate(s) on plane (equivalent to l), in radians
    y_rad : float or array
        Elevation-like coordinate(s) on plane (equivalent to m), in radians
    """
    wcs = _prepare_wcs(code)
    azel0_deg = np.rad2deg(np.c_[az0_rad, el0_rad])
    azel_deg = np.rad2deg(np.c_[az_rad, el_rad])
    # WCSLIB supports vector (az, el) but only scalar (az0, el0)
    if azel0_deg.shape[0] == 1:
        wcs.wcs.crval = azel0_deg[0]
        xy_deg = wcs.wcs.s2p(azel_deg, origin=1)["imgcrd"]
    else:
        xy_deg = np.zeros_like(azel_deg)
        for n in range(azel_deg.shape[0]):
            wcs.wcs.crval = azel0_deg[n]
            xy_deg[n] = wcs.wcs.s2p(azel_deg[n : n + 1], origin=1)["imgcrd"]
    xy_rad = np.deg2rad(xy_deg)
    # XXX Support out-of-range treatment (at least "nan" and "raise") via s2p "stat"
    return _fix_scalar(xy_rad[:, 0], az_rad), _fix_scalar(xy_rad[:, 1], el_rad)


def plane_to_sphere_wcs(code, az0_rad, el0_rad, x_rad, y_rad):
    """Deproject plane to sphere using specified WCSLIB projection.

    Parameters
    ----------
    code : str
        Three-letter projection code defined by the FITS standard
    az0_rad : float or array
        Azimuth / right ascension / longitude of reference point(s), in radians
    el0_rad : float or array
        Elevation / declination / latitude of reference point(s), in radians
    x_rad : float or array
        Azimuth-like coordinate(s) on plane (equivalent to l), in radians
    y_rad : float or array
        Elevation-like coordinate(s) on plane (equivalent to m), in radians

    Returns
    -------
    az_rad : float or array
        Azimuth / right ascension / longitude of deprojected point(s), in radians
    el_rad : float or array
        Elevation / declination / latitude of deprojected point(s), in radians
    """
    wcs = _prepare_wcs(code)
    azel0_deg = np.rad2deg(np.c_[az0_rad, el0_rad])
    xy_deg = np.rad2deg(np.c_[x_rad, y_rad])
    # WCSLIB supports vector (x, y) but only scalar (az0, el0)
    if azel0_deg.shape[0] == 1:
        wcs.wcs.crval = azel0_deg[0]
        azel_deg = wcs.wcs.p2s(xy_deg, origin=1)["world"]
    else:
        azel_deg = np.zeros_like(xy_deg)
        for n in range(xy_deg.shape[0]):
            wcs.wcs.crval = azel0_deg[n]
            azel_deg[n] = wcs.wcs.p2s(xy_deg[n : n + 1], origin=1)["world"]
    azel_rad = np.deg2rad(azel_deg)
    # XXX Support out-of-range treatment (at least "nan" and "raise") via p2s "stat"
    return _fix_scalar(azel_rad[:, 0], x_rad), _fix_scalar(azel_rad[:, 1], y_rad)


# The decimal accuracy for each projection is the maximum that makes
# the test pass during an extended random run (N = 100000).
# CAR has issues (see e.g. https://github.com/astropy/astropy/issues/6490).
@pytest.mark.parametrize(
    "projection,decimal",
    [("SIN", 11), ("TAN", 11), ("ARC", 11), ("STG", 11)],
)
def test_wcs_compatibility(random, projection, decimal, N=100):
    """Compare to corresponding WCSLIB routine."""
    plane_to_sphere = katpoint.plane_to_sphere[projection]
    sphere_to_plane = katpoint.sphere_to_plane[projection]
    az0, el0, x, y = generate_data[projection](random, N)
    az, el = plane_to_sphere(az0, el0, x, y)
    xx, yy = sphere_to_plane(az0, el0, az, el)
    az_wcs, el_wcs = plane_to_sphere_wcs(projection, az0, el0, x, y)
    x_wcs, y_wcs = sphere_to_plane_wcs(projection, az0, el0, az, el)
    assert_angles_almost_equal(az, az_wcs, decimal=decimal)
    assert_angles_almost_equal(el, el_wcs, decimal=decimal)
    np.testing.assert_almost_equal(xx, x_wcs, decimal=decimal)
    np.testing.assert_almost_equal(yy, y_wcs, decimal=decimal)


@pytest.mark.parametrize(
    "projection, sphere, plane",
    [
        # Reference point at pole on sphere
        ("SSN", (0.0, PI / 2, 0.0, 0.0), [0.0, 1.0]),
        ("SSN", (0.0, PI / 2, PI, 1e-12), [0.0, 1.0]),
        ("SSN", (0.0, PI / 2, PI / 2, 0.0), [0.0, 1.0]),
        ("SSN", (0.0, PI / 2, -PI / 2, 0.0), [0.0, 1.0]),
    ],
)
def test_sphere_to_plane(projection, sphere, plane, decimal=12):
    """Test specific cases (sphere -> plane)."""
    sphere_to_plane = katpoint.sphere_to_plane[projection]
    xy = np.array(sphere_to_plane(*sphere))
    np.testing.assert_almost_equal(xy, plane, decimal)


def sphere_to_plane_invalid(projection, sphere, clipped, decimal):
    """Test points outside allowed domain on sphere (sphere -> plane)."""
    sphere_to_plane = katpoint.sphere_to_plane[projection]
    with out_of_range_context("raise"):
        with pytest.raises(OutOfRangeError):
            sphere_to_plane(*sphere)
    with out_of_range_context("nan"):
        np.testing.assert_array_equal(sphere_to_plane(*sphere), [np.nan, np.nan])
    with out_of_range_context("clip"):
        test_sphere_to_plane(projection, sphere, clipped, decimal)


@pytest.mark.parametrize(
    "projection, clip_x, clip_y, decimal",
    [
        ("SIN", -1.0, 1.0, 12),
        ("TAN", -1e6, 1e6, 4),
        ("ARC", np.nan, PI / 2, 12),
        ("STG", -894.42495493, 2.0, 8),
        ("SSN", -1.0, -1.0, 12),
    ],
)
def test_sphere_to_plane_outside_domain(projection, clip_x, clip_y, decimal):
    """Test points outside allowed domain on sphere (sphere -> plane)."""
    sphere_to_plane_invalid(projection, (0.0, PI, 0.0, 0.0), [0.0, -clip_y], decimal)
    sphere_to_plane_invalid(projection, (0.0, 0.0, 0.0, PI), [0.0, +clip_y], decimal)
    if projection != "ARC":
        sphere_to_plane_invalid(projection, (0.0, 0.0, PI, 0.0), [clip_x, 0.0], decimal)


def test_sphere_to_plane_special():
    """Test special corner cases (sphere -> plane)."""
    sphere_to_plane = katpoint.sphere_to_plane["ARC"]
    # Point diametrically opposite the reference point on sphere
    xy = np.array(sphere_to_plane(PI, 0.0, 0.0, 0.0))
    np.testing.assert_almost_equal(np.abs(xy), [PI, 0.0], decimal=12)


@pytest.mark.parametrize(
    "projection, plane, sphere",
    [
        # Points on circle with radius pi in plane
        ("ARC", (0.0, 0.0, PI, 0.0), [PI, 0.0]),
        ("ARC", (0.0, 0.0, -PI, 0.0), [-PI, 0.0]),
        ("ARC", (0.0, 0.0, 0.0, PI), [PI, 0.0]),
        ("ARC", (0.0, 0.0, 0.0, -PI), [PI, 0.0]),
        # Reference point at pole on sphere
        ("SSN", (0.0, PI / 2, 0.0, 1.0), [0.0, 0.0]),
        ("SSN", (0.0, -PI / 2, 0.0, -1.0), [0.0, 0.0]),
        # Test valid (x, y) domain
        ("SSN", (0.0, 1.0, 0.0, -np.cos(1.0)), [0.0, PI / 2]),
        ("SSN", (0.0, -1.0, 0.0, np.cos(1.0)), [0.0, -PI / 2]),
    ],
)
def test_plane_to_sphere(projection, plane, sphere):
    """Test specific cases (plane -> sphere)."""
    plane_to_sphere = katpoint.plane_to_sphere[projection]
    ae = np.array(plane_to_sphere(*plane))
    assert_angles_almost_equal(ae, sphere, decimal=12)


def plane_to_sphere_invalid(projection, plane, clipped):
    """Test points outside allowed domain in plane (plane -> sphere)."""
    plane_to_sphere = katpoint.plane_to_sphere[projection]
    with out_of_range_context("raise"):
        with pytest.raises(OutOfRangeError):
            plane_to_sphere(*plane)
    with out_of_range_context("nan"):
        np.testing.assert_array_equal(plane_to_sphere(*plane), [np.nan, np.nan])
    with out_of_range_context("clip"):
        test_plane_to_sphere(projection, plane, clipped)


@pytest.mark.parametrize(
    "projection, offset_p",
    [("SIN", 2.0), ("TAN", np.nan), ("ARC", 4.0), ("STG", np.nan), ("SSN", -2.0)],
)
def test_plane_to_sphere_outside_domain(projection, offset_p):
    """Test points outside allowed domain in plane (plane -> sphere)."""
    # Bad el0 > 90 degrees
    plane_to_sphere_invalid(projection, (0.0, PI, 0.0, 0.0), [0.0, PI / 2])
    if projection == "ARC":
        plane_to_sphere_invalid(projection, (0.0, 0.0, offset_p, 0.0), [PI, 0.0])
        plane_to_sphere_invalid(projection, (0.0, 0.0, 0.0, offset_p), [PI, 0.0])
    elif not np.isnan(offset_p):
        # Bad (x, y) vector length > 1.0
        plane_to_sphere_invalid(projection, (0.0, 0.0, offset_p, 0.0), [PI / 2, 0.0])
        plane_to_sphere_invalid(projection, (0.0, 0.0, 0.0, offset_p), [0.0, PI / 2])
    if projection == "SSN":
        # Bad x coordinate > cos(el0)
        plane_to_sphere_invalid(projection, (0.0, PI / 2, 1.0, 0.0), [-PI / 2, 0.0])
        plane_to_sphere_invalid(projection, (0.0, PI / 2, -1.0, 0.0), [PI / 2, 0.0])
        # Bad y coordinate -> den < 0
        plane_to_sphere_invalid(projection, (0.0, PI / 2, 0.0, -1.0), [0.0, PI / 2])


def sphere_to_plane_to_sphere(projection, reference, sphere, plane):
    """Project from sphere to plane and back again and check results on both legs."""
    test_sphere_to_plane(projection, tuple(reference) + tuple(sphere), plane)
    test_plane_to_sphere(projection, tuple(reference) + tuple(plane), sphere)


@pytest.mark.parametrize(
    "projection, offset_s, offset_p",
    [
        ("SIN", PI / 2, 1.0),
        ("TAN", PI / 4, 1.0),
        ("ARC", PI / 2, PI / 2),
        ("STG", PI / 2, 2.0),
        ("SSN", PI / 2, -1.0),
    ],
)
def test_sphere_to_plane_to_sphere_origin(projection, offset_s, offset_p):
    """Test five-point cross along axes, centred on origin (sphere->plane->sphere)."""
    sphere_to_plane_to_sphere(projection, (0.0, 0.0), (0.0, 0.0), [0.0, 0.0])
    sphere_to_plane_to_sphere(
        projection, (0.0, 0.0), (+offset_s, 0.0), [+offset_p, 0.0]
    )
    sphere_to_plane_to_sphere(
        projection, (0.0, 0.0), (-offset_s, 0.0), [-offset_p, 0.0]
    )
    sphere_to_plane_to_sphere(
        projection, (0.0, 0.0), (0.0, +offset_s), [0.0, +offset_p]
    )
    sphere_to_plane_to_sphere(
        projection, (0.0, 0.0), (0.0, -offset_s), [0.0, -offset_p]
    )


@pytest.mark.parametrize(
    "projection, offset_s, offset_p",
    [
        ("SIN", PI / 2 - 1e-12, 1.0),
        ("TAN", PI / 4, 1.0),
        ("ARC", PI / 2, PI / 2),
        ("STG", PI / 2, 2.0),
    ],
)
def test_sphere_to_plane_to_sphere_pole(projection, offset_s, offset_p):
    """Test four-point cross along axes, centred on pole (sphere -> plane -> sphere)."""
    el = PI / 2 - offset_s
    sphere_to_plane_to_sphere(
        projection, (0.0, PI / 2), (+PI / 2, el), [+offset_p, 0.0]
    )
    sphere_to_plane_to_sphere(
        projection, (0.0, PI / 2), (-PI / 2, el), [-offset_p, 0.0]
    )
    sphere_to_plane_to_sphere(projection, (0.0, PI / 2), (PI, el), [0.0, +offset_p])
    sphere_to_plane_to_sphere(projection, (0.0, PI / 2), (0.0, el), [0.0, -offset_p])


def sphere_to_plane_original_ssn(target_az, target_el, scan_az, scan_el):
    """Mattieu's original version of SSN projection."""
    ll = np.cos(target_el) * np.sin(target_az - scan_az)
    mm = np.cos(target_el) * np.sin(scan_el) * np.cos(target_az - scan_az) - np.cos(
        scan_el
    ) * np.sin(target_el)
    return ll, mm


def plane_to_sphere_original_ssn(target_az, target_el, ll, mm):
    """Mattieu's original version of SSN projection."""
    scan_az = target_az - np.arcsin(np.clip(ll / np.cos(target_el), -1.0, 1.0))
    scan_el = np.arcsin(
        np.clip(
            (
                np.sqrt(1.0 - ll**2 - mm**2) * np.sin(target_el)
                + np.sqrt(np.cos(target_el) ** 2 - ll**2) * mm
            )
            / (1.0 - ll**2),
            -1.0,
            1.0,
        )
    )
    return scan_az, scan_el


def test_vs_original_ssn(random, decimal=10, N=100):
    """SSN projection: compare against Mattieu's original version."""
    plane_to_sphere = katpoint.plane_to_sphere["SSN"]
    az0, el0, x, y = generate_data["SSN"](random, N)
    az, el = plane_to_sphere(az0, el0, x, y)
    ll, mm = sphere_to_plane_original_ssn(az0, el0, az, el)
    aa, ee = plane_to_sphere_original_ssn(az0, el0, ll, mm)
    np.testing.assert_almost_equal(x, ll, decimal=decimal)
    np.testing.assert_almost_equal(y, -mm, decimal=decimal)
    assert_angles_almost_equal(az, aa, decimal=decimal)
    assert_angles_almost_equal(el, ee, decimal=decimal)
