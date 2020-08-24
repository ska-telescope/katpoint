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

"""Tests for the target module."""

import time
import pickle
from contextlib import contextmanager

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import Angle

import katpoint
from katpoint.test.helper import check_separation


# Use the current year in TLE epochs to avoid potential crashes due to expired TLEs
YY = time.localtime().tm_year % 100
TLE_TARGET = ('tle, GPS BIIA-21 (PRN 09)    \n'
              '1 22700U 93042A   {:02d}266.32333151  .00000012  00000-0  10000-3 0  805{:1d}\n'
              '2 22700  55.4408  61.3790 0191986  78.1802 283.9935  2.00561720104282\n'
              .format(YY, (YY // 10 + YY - 7 + 4) % 10))


class TestTargetConstruction:
    """Test construction of targets from strings and vice versa."""

    def setup(self):
        self.azel_target = 'azel, 10.0, -10.0'
        # A floating-point RA is in degrees
        self.radec_target = 'radec, 20.0, -20.0'
        # A sexagesimal RA string is in hours
        self.radec_target_rahours = 'radec, 20:00:00, -20:00:00'
        self.gal_target = 'gal, 30.0, -30.0'
        self.tag_target = 'azel J2000 GPS, 40.0, -30.0'

    def test_construct_target(self):
        """Test construction of targets from strings and vice versa."""
        azel1 = katpoint.Target(self.azel_target)
        azel2 = katpoint.construct_azel_target('10:00:00.0', '-10:00:00.0')
        assert azel1 == azel2, 'Special azel constructor failed'
        radec1 = katpoint.Target(self.radec_target)
        radec2 = katpoint.construct_radec_target('20.0', '-20.0')
        assert radec1 == radec2, 'Special radec constructor (decimal) failed'
        radec3 = katpoint.Target(self.radec_target_rahours)
        radec4 = katpoint.construct_radec_target('20:00:00.0', '-20:00:00.0')
        assert radec3 == radec4, 'Special radec constructor (sexagesimal) failed'
        radec5 = katpoint.construct_radec_target('20:00:00.0', '-00:30:00.0')
        radec6 = katpoint.construct_radec_target('300.0', '-0.5')
        assert radec5 == radec6, 'Special radec constructor (decimal <-> sexagesimal) failed'
        # Check that description string updates when object is updated
        t1 = katpoint.Target('piet, azel, 20, 30')
        t2 = katpoint.Target('piet | bollie, azel, 20, 30')
        assert t1 != t2, 'Targets should not be equal'
        t1.aliases += ['bollie']
        assert t1.description == t2.description, 'Target description string not updated'
        assert t1 == t2.description, 'Equality with description string failed'
        assert t1 == t2, 'Equality with target failed'
        assert t1 == katpoint.Target(t2), 'Construction with target object failed'
        assert t1 == pickle.loads(pickle.dumps(t1)), 'Pickling failed'
        try:
            assert hash(t1) == hash(t2), 'Target hashes not equal'
        except TypeError:
            pytest.fail('Target object not hashable')

    def test_constructed_coords(self):
        """Test whether calculated coordinates match those with which it is constructed."""
        # azel
        azel = katpoint.Target(self.azel_target)
        calc_azel = azel.azel()
        assert calc_azel.az.deg == 10.0
        assert calc_azel.alt.deg == -10.0
        # radec (degrees)
        radec = katpoint.Target(self.radec_target)
        calc_radec = radec.radec()
        assert calc_radec.ra.deg == 20.0
        assert calc_radec.dec.deg == -20.0
        # radec (hours)
        radec_rahours = katpoint.Target(self.radec_target_rahours)
        calc_radec_rahours = radec_rahours.radec()
        assert calc_radec_rahours.ra.hms == (20, 0, 0)
        assert calc_radec_rahours.dec.deg == -20.0
        # gal
        lb = katpoint.Target(self.gal_target)
        calc_lb = lb.galactic()
        assert calc_lb.l.deg == 30.0
        assert calc_lb.b.deg == -30.0

    def test_add_tags(self):
        """Test adding tags."""
        tag_target = katpoint.Target(self.tag_target)
        tag_target.add_tags(None)
        tag_target.add_tags('pulsar')
        tag_target.add_tags(['SNR', 'GPS'])
        assert tag_target.tags == ['azel', 'J2000', 'GPS', 'pulsar', 'SNR'], (
            'Added tags not correct')


@pytest.mark.parametrize(
    "description",
    [
        'azel, -30.0, 90.0',
        ', azel, 180, -45:00:00.0',
        'Zenith, azel, 0, 90',
        'radec J2000, 0, 0.0, (1000.0 2000.0 1.0 10.0)',
        ', radec B1950, 14:23:45.6, -60:34:21.1',
        'radec B1900, 14:23:45.6, -60:34:21.1',
        'gal, 300.0, 0.0',
        'Sag A, gal, 0.0, 0.0',
        'Zizou, radec cal, 1.4, 30.0, (1000.0 2000.0 1.0 10.0)',
        'Fluffy | *Dinky, radec, 12.5, -50.0, (1.0 2.0 1.0 2.0 3.0 4.0)',
        TLE_TARGET,
        ', ' + TLE_TARGET,
        'Sun, special',
        'Nothing, special',
        'Moon | Luna, special solarbody',
        'Aldebaran, star',
        'Betelgeuse | Maitland, star orion',
        'xephem star, Sadr~f|S|F8~20:22:13.7|2.43~40:15:24|-0.93~2.23~2000~0',
        'Acamar | Theta Eridani, xephem, HIC 13847~f|S|A4~2:58:16.03~-40:18:17.1~2.906~2000~0',
        'Kakkab, xephem, H71860 | S225128~f|S|B1~14:41:55.768~-47:23:17.51~2.304~2000~0',
    ]
)
def test_construct_valid_target(description):
    """Test construction of valid targets from strings and vice versa."""
    # Normalise description string through one cycle to allow comparison
    reference_description = katpoint.Target(description).description
    test_target = katpoint.Target(reference_description)
    assert test_target.description == reference_description, (
        "Target description ('{}') differs from reference ('{}')"
        .format(test_target.description, reference_description))
    # Exercise repr() and str()
    print('{!r} {}'.format(test_target, test_target))


@pytest.mark.parametrize(
    "description",
    [
        'Sun',
        'Sun, ',
        '-30.0, 90.0',
        ', azel, -45:00:00.0',
        'Zenith, azel blah',
        'radec J2000, 0.3',
        'gal, 0.0',
        'Zizou, radec cal, 1.4, 30.0, (1000.0, 2000.0, 1.0, 10.0)',
        # TLE missing the first line
        ('tle, GPS BIIA-21 (PRN 09)    \n'
         '2 22700  55.4408  61.3790 0191986  78.1802 283.9935  2.00561720104282\n'),
        # TLE missing the satellite catalog number and classification on line 1
        (', tle, GPS BIIA-22 (PRN 05)    \n'
         '1 93054A   {:02d}266.92814765  .00000062  00000-0  10000-3 0  289{:1d}\n'
         '2 22779  53.8943 118.4708 0081407  68.2645 292.7207  2.00558015103055\n'
         .format(YY, (YY // 10 + YY - 7 + 5) % 10)),
        'Sunny, special',
        'Slinky, star',
        'xephem star, Sadr~20:22:13.7|2.43~40:15:24|-0.93~2.23~2000~0',
        'hotbody, 34.0, 45.0',
    ]
)
def test_construct_invalid_target(description):
    """Test construction of invalid targets from strings."""
    with pytest.raises(ValueError):
        katpoint.Target(description)


NON_AZEL = 'astrometric_radec apparent_radec galactic'


@contextmanager
def does_not_raise(error):
    yield


@pytest.mark.parametrize(
    "description,methods,raises,error",
    [
        ('azel, 10, -10', 'azel', does_not_raise, None),
        ('azel, 10, -10', NON_AZEL, pytest.raises, ValueError),
        ('radec, 20, -20', 'azel', pytest.raises, ValueError),
        ('radec, 20, -20', NON_AZEL, does_not_raise, None),
        ('gal, 30, -30', 'azel', pytest.raises, ValueError),
        ('gal, 30, -30', NON_AZEL, does_not_raise, None),
        ('Sun, special', 'azel', pytest.raises, ValueError),
        ('Sun, special', NON_AZEL, does_not_raise, None),
        (TLE_TARGET, 'azel', pytest.raises, ValueError),
        (TLE_TARGET, NON_AZEL, does_not_raise, None),
    ]
)
def test_coord_methods_without_antenna(description, methods, raises, error):
    """"Test whether coordinate methods can operate without an Antenna."""
    target = katpoint.Target(description)
    for method in methods.split():
        with raises(error):
            getattr(target, method)()


TARGET = katpoint.construct_azel_target('45:00:00.0', '75:00:00.0')
ANT1 = katpoint.Antenna('A1, -31.0, 18.0, 0.0, 12.0, 0.0 0.0 0.0')
ANT2 = katpoint.Antenna('A2, -31.0, 18.0, 0.0, 12.0, 10.0 -10.0 0.0')
TS = katpoint.Timestamp('2013-08-14 09:25')


def _array_vs_scalar(func, array_in, sky_coord=False, pre_shape=(), post_shape=()):
    """Check that `func` output for ndarray of inputs is array of corresponding scalar outputs."""
    array_out = func(array_in)
    assert np.shape(array_out) == pre_shape + array_in.shape + post_shape
    all_pre = len(pre_shape) * (np.s_[:],)
    all_post = len(post_shape) * (np.s_[:],)
    for index_in in np.ndindex(array_in.shape):
        scalar = func(array_in[index_in])
        if sky_coord:
            # Treat output as if it is SkyCoord with internal array, check separation instead
            assert array_out[index_in].separation(scalar).rad == pytest.approx(0.0)
        else:
            # Assume that function outputs more complicated ndarrays of numbers (or equivalent)
            array_slice = np.asarray(array_out)[all_pre + index_in + all_post]
            np.testing.assert_array_equal(array_slice, np.asarray(scalar))


@pytest.mark.parametrize("description", ['azel, 10, -10', 'radec, 20, -20',
                                         'gal, 30, -30', 'Sun, special', TLE_TARGET])
def test_array_valued_methods(description):
    """Test array-valued methods, comparing output against corresponding scalar versions."""
    offsets = np.array([[[0, 1, 2, 3], [4, 5, 6, 7]]])
    times = (katpoint.Timestamp('2020-07-30 14:02:00') + offsets).time
    assert times.shape == offsets.shape
    target = katpoint.Target(description)
    _array_vs_scalar(lambda t: target.azel(t, ANT1), times, sky_coord=True)
    _array_vs_scalar(lambda t: target.apparent_radec(t, ANT1), times, sky_coord=True)
    _array_vs_scalar(lambda t: target.astrometric_radec(t, ANT1), times, sky_coord=True)
    _array_vs_scalar(lambda t: target.galactic(t, ANT1), times, sky_coord=True)
    _array_vs_scalar(lambda t: target.parallactic_angle(t, ANT1), times)
    _array_vs_scalar(lambda t: target.geometric_delay(ANT2, t, ANT1), times, pre_shape=(2,))
    _array_vs_scalar(lambda t: target.uvw_basis(t, ANT1), times, pre_shape=(3, 3))
    _array_vs_scalar(lambda t: target.uvw([ANT1, ANT2], t, ANT1),
                     times, pre_shape=(3,), post_shape=(2,))
    _array_vs_scalar(lambda t: target.lmn(0.0, 0.0, t, ANT1), times, pre_shape=(3,))
    l, m, n = target.lmn(np.zeros_like(offsets), np.zeros_like(offsets), times, ANT1)
    assert l.shape == m.shape == n.shape == offsets.shape
    np.testing.assert_allclose(target.separation(target, times, ANT1).rad,
                               np.zeros_like(offsets), atol=1e-12)


def test_coords():
    """Test coordinate conversions for coverage and verification."""
    coord = TARGET.azel(TS, ANT1)
    assert coord.az.deg == 45  # PyEphem: 45
    assert coord.alt.deg == 75  # PyEphem: 75
    coord = TARGET.apparent_radec(TS, ANT1)
    check_separation(coord, '8:53:03.49166920h', '-19:54:51.92328722d', tol=1 * u.mas)
    # PyEphem:               8:53:09.60,          -19:51:43.0 (same as astrometric)
    coord = TARGET.astrometric_radec(TS, ANT1)
    check_separation(coord, '8:53:09.60397465h', '-19:51:42.87773802d', tol=1 * u.mas)
    # PyEphem:               8:53:09.60,          -19:51:43.0
    coord = TARGET.galactic(TS, ANT1)
    check_separation(coord, '245:34:49.20442837d', '15:36:24.87974969d', tol=1 * u.mas)
    # PyEphem:               245:34:49.3,           15:36:24.7
    coord = TARGET.parallactic_angle(TS, ANT1)
    assert coord.deg == pytest.approx(-140.279593566336)  # PyEphem: -140.34440985011398


DELAY_TARGET = katpoint.Target('radec, 20.0, -20.0')
DELAY_TS = [TS, TS + 1.0]
DELAY = [1.75538294e-08, 1.75522002e-08]
DELAY_RATE = [-1.62915174e-12, -1.62929689e-12]
UVW = ([-7.118580813334029, -11.028682662045913, -5.262505671628351],
       [-7.119215642091996, -11.028505936045280, -5.262017242465739])


def test_delay():
    """Test geometric delay."""
    delay, delay_rate = DELAY_TARGET.geometric_delay(ANT2, DELAY_TS[0], ANT1)
    np.testing.assert_allclose(delay, DELAY[0])
    np.testing.assert_allclose(delay_rate, DELAY_RATE[0])
    delay, delay_rate = DELAY_TARGET.geometric_delay(ANT2, DELAY_TS, ANT1)
    np.testing.assert_allclose(delay, DELAY)
    np.testing.assert_allclose(delay_rate, DELAY_RATE)


def test_uvw():
    """Test uvw calculation."""
    u, v, w = DELAY_TARGET.uvw(ANT2, DELAY_TS[0], ANT1)
    np.testing.assert_almost_equal([u, v, w], UVW[0], decimal=8)
    u, v, w = DELAY_TARGET.uvw(ANT2, DELAY_TS, ANT1)
    np.testing.assert_array_almost_equal([u, v, w], np.c_[UVW], decimal=8)


def test_uvw_timestamp_array_azel():
    """Test uvw calculation on a timestamp array when the target is an azel target."""
    azel = DELAY_TARGET.azel(DELAY_TS[0], ANT1)
    target = katpoint.construct_azel_target(azel.az, azel.alt)
    u, v, w = target.uvw(ANT2, DELAY_TS, ANT1)
    np.testing.assert_array_almost_equal([u[0], v[0], w[0]], UVW[0], decimal=8)
    np.testing.assert_array_almost_equal(w, [UVW[0][2]] * len(DELAY_TS), decimal=8)


def test_uvw_antenna_array():
    u, v, w = DELAY_TARGET.uvw([ANT1, ANT2], DELAY_TS[0], ANT1)
    np.testing.assert_array_almost_equal([u, v, w], np.c_[np.zeros(3), UVW[0]], decimal=8)


def test_uvw_both_array():
    u, v, w = DELAY_TARGET.uvw([ANT1, ANT2], DELAY_TS, ANT1)
    # UVW array has shape (3, n_times, n_bls) - stack times along dim 1 and ants along dim 2
    desired_uvw = np.dstack([np.zeros((3, len(DELAY_TS))), np.c_[UVW]])
    np.testing.assert_array_almost_equal([u, v, w], desired_uvw, decimal=8)


def test_uvw_hemispheres():
    """Test uvw calculation near the equator.

    The implementation behaves differently depending on the sign of
    declination. This test is to catch sign flip errors.
    """
    target1 = katpoint.construct_radec_target(0.0, -1e-9)
    target2 = katpoint.construct_radec_target(0.0, +1e-9)
    u1, v1, w1 = target1.uvw(ANT2, TS, ANT1)
    u2, v2, w2 = target2.uvw(ANT2, TS, ANT1)
    np.testing.assert_almost_equal(u1, u2, decimal=3)
    np.testing.assert_almost_equal(v1, v2, decimal=3)
    np.testing.assert_almost_equal(w1, w2, decimal=3)


def test_lmn():
    """Test lmn calculation."""
    # For angles less than pi/2, it matches SIN projection
    pointing = katpoint.construct_radec_target('11:00:00.0', '-75:00:00.0')
    target = katpoint.construct_radec_target('16:00:00.0', '-65:00:00.0')
    radec = target.radec(timestamp=TS, antenna=ANT1)
    l, m, n = pointing.lmn(radec.ra.rad, radec.dec.rad)
    expected_l, expected_m = pointing.sphere_to_plane(
        radec.ra.rad, radec.dec.rad, projection_type='SIN', coord_system='radec')
    expected_n = np.sqrt(1.0 - expected_l**2 - expected_m**2)
    np.testing.assert_almost_equal(l, expected_l, decimal=12)
    np.testing.assert_almost_equal(m, expected_m, decimal=12)
    np.testing.assert_almost_equal(n, expected_n, decimal=12)
    # Test angle > pi/2: using the diametrically opposite target
    l, m, n = pointing.lmn(np.pi + radec.ra.rad, -radec.dec.rad)
    np.testing.assert_almost_equal(l, -expected_l, decimal=12)
    np.testing.assert_almost_equal(m, -expected_m, decimal=12)
    np.testing.assert_almost_equal(n, -expected_n, decimal=12)


def test_separation():
    """Test separation calculation."""
    sun = katpoint.Target('Sun, special')
    azel_sun = sun.azel(TS, ANT1)
    azel = katpoint.construct_azel_target(azel_sun.az, azel_sun.alt)
    sep = sun.separation(azel, TS, ANT1)
    np.testing.assert_almost_equal(sep.rad, 0.0)
    sep = azel.separation(sun, TS, ANT1)
    np.testing.assert_almost_equal(sep.rad, 0.0)
    azel2 = katpoint.construct_azel_target(azel_sun.az,
                                           azel_sun.alt + Angle(0.01, unit=u.rad))
    sep = azel.separation(azel2, TS, ANT1)
    np.testing.assert_almost_equal(sep.rad, 0.01, decimal=12)


def test_projection():
    """Test projection."""
    az, el = katpoint.deg2rad(50.0), katpoint.deg2rad(80.0)
    x, y = TARGET.sphere_to_plane(az, el, TS, ANT1)
    re_az, re_el = TARGET.plane_to_sphere(x, y, TS, ANT1)
    np.testing.assert_almost_equal(re_az, az, decimal=12)
    np.testing.assert_almost_equal(re_el, el, decimal=12)
