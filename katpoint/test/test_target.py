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
        calc_az = calc_azel.az
        calc_el = calc_azel.alt
        assert calc_az.deg == 10.0
        assert calc_el.deg == -10.0
        # radec (degrees)
        radec = katpoint.Target(self.radec_target)
        calc_radec = radec.radec()
        calc_ra = calc_radec.ra
        calc_dec = calc_radec.dec
        assert calc_ra.deg == 20.0
        assert calc_dec.deg == -20.0
        # radec (hours)
        radec_rahours = katpoint.Target(self.radec_target_rahours)
        calc_radec_rahours = radec_rahours.radec()
        calc_ra = calc_radec_rahours.ra
        calc_dec = calc_radec_rahours.dec
        assert calc_ra.hms == (20, 0, 0)
        assert calc_dec.deg == -20.0
        # gal
        lb = katpoint.Target(self.gal_target)
        calc_lb = lb.galactic()
        calc_l = calc_lb.l
        calc_b = calc_lb.b
        assert calc_l.deg == 30.0
        assert calc_b.deg == -30.0

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
        (', tle, GPS BIIA-22 (PRN 05)    \n'
         '1 22779U 93054A   {:02d}266.92814765  .00000062  00000-0  10000-3 0  289{:1d}\n'
         '2 22779  53.8943 118.4708 0081407  68.2645 292.7207  2.00558015103055\n'
         .format(YY, (YY // 10 + YY - 7 + 5) % 10)),
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
        ('tle, GPS BIIA-21 (PRN 09)    \n'
         '2 22700  55.4408  61.3790 0191986  78.1802 283.9935  2.00561720104282\n'),
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
        (TLE_TARGET, 'azel ' + NON_AZEL, pytest.raises, ValueError),
    ]
)
def test_coord_methods_without_antenna(description, methods, raises, error):
    """"Test whether coordinate methods can operate without an Antenna."""
    target = katpoint.Target(description)
    for method in methods.split():
        with raises(error):
            getattr(target, method)()


# XXX TLE_TARGET does not support array timestamps yet
@pytest.mark.parametrize("description", ['azel, 10, -10', 'radec, 20, -20',
                                         'gal, 30, -30', 'Sun, special'])
def test_array_valued_azel(description):
    """Test array-valued (az, el) coordinates."""
    ts = katpoint.Timestamp('2020-07-30 14:02:00')
    ant1 = katpoint.Antenna('A1, -31.0, 18.0, 0.0, 12.0, 0.0 0.0 0.0')
    offsets = np.array([np.arange(3), np.arange(3)])
    times = ts + offsets
    assert times.time.shape == offsets.shape
    target = katpoint.Target(description)
    assert target.azel(times, ant1).shape == offsets.shape
    assert target.astrometric_radec(times, ant1).shape == offsets.shape
    assert target.apparent_radec(times, ant1).shape == offsets.shape
    assert target.galactic(times, ant1).shape == offsets.shape
    assert target.parallactic_angle(times, ant1).shape == offsets.shape
    assert target.separation(target, times, ant1).shape == offsets.shape


class TestTargetCalculations:
    """Test various calculations involving antennas and timestamps."""

    def setup(self):
        self.target = katpoint.construct_azel_target('45:00:00.0', '75:00:00.0')
        self.ant1 = katpoint.Antenna('A1, -31.0, 18.0, 0.0, 12.0, 0.0 0.0 0.0')
        self.ant2 = katpoint.Antenna('A2, -31.0, 18.0, 0.0, 12.0, 10.0 -10.0 0.0')
        self.ts = katpoint.Timestamp('2013-08-14 09:25')
        # self.uvw = [10.822861713680807, -9.103057965680664, -2.220446049250313e-16]
        self.uvw = [10.820796672358002, -9.1055125816993954, -2.22044604925e-16]

    def test_coords(self):
        """Test coordinate conversions for coverage and verification."""
        coord = self.target.azel(self.ts, self.ant1)
        assert coord.az.deg == 45  # PyEphem: 45
        assert coord.alt.deg == 75  # PyEphem: 75
        coord = self.target.apparent_radec(self.ts, self.ant1)
        ra_hour = coord.ra.to_string(unit='hour', sep=':', precision=8)
        dec_deg = coord.dec.to_string(sep=':', precision=8)
        assert ra_hour == '8:53:03.49166920'  # PyEphem: 8:53:09.60 (same as astrometric)
        assert dec_deg == '-19:54:51.92328722'  # PyEphem: -19:51:43.0 (same as astrometric)
        coord = self.target.astrometric_radec(self.ts, self.ant1)
        ra_hour = coord.ra.to_string(unit='hour', sep=':', precision=8)
        dec_deg = coord.dec.to_string(sep=':', precision=8)
        assert ra_hour == '8:53:09.60397465'  # PyEphem: 8:53:09.60
        assert dec_deg == '-19:51:42.87773802'  # PyEphem: -19:51:43.0
        coord = self.target.galactic(self.ts, self.ant1)
        l_deg = coord.l.to_string(sep=':', precision=8)
        b_deg = coord.b.to_string(sep=':', precision=8)
        assert l_deg == '245:34:49.20442837'  # PyEphem: 245:34:49.3
        assert b_deg == '15:36:24.87974969'  # PyEphem: 15:36:24.7
        coord = self.target.parallactic_angle(self.ts, self.ant1)
        assert coord.deg == pytest.approx(-140.279593566336)  # PyEphem: -140.34440985011398

    def test_delay(self):
        """Test geometric delay."""
        delay, delay_rate = self.target.geometric_delay(self.ant2, self.ts, self.ant1)
        np.testing.assert_almost_equal(delay, 0.0, decimal=12)
        np.testing.assert_almost_equal(delay_rate, 0.0, decimal=12)
        delay, delay_rate = self.target.geometric_delay(self.ant2, [self.ts, self.ts], self.ant1)
        np.testing.assert_almost_equal(delay, np.array([0.0, 0.0]), decimal=12)
        np.testing.assert_almost_equal(delay_rate, np.array([0.0, 0.0]), decimal=12)

    def test_uvw(self):
        """Test uvw calculation."""
        u, v, w = self.target.uvw(self.ant2, self.ts, self.ant1)
        np.testing.assert_almost_equal(u, self.uvw[0], decimal=5)
        np.testing.assert_almost_equal(v, self.uvw[1], decimal=5)
        np.testing.assert_almost_equal(w, self.uvw[2], decimal=5)

    def test_uvw_timestamp_array(self):
        """Test uvw calculation on an array."""
        u, v, w = self.target.uvw(self.ant2, np.array([self.ts, self.ts]), self.ant1)
        np.testing.assert_array_almost_equal(u, np.array([self.uvw[0]] * 2), decimal=5)
        np.testing.assert_array_almost_equal(v, np.array([self.uvw[1]] * 2), decimal=5)
        np.testing.assert_array_almost_equal(w, np.array([self.uvw[2]] * 2), decimal=5)

    def test_uvw_timestamp_array_radec(self):
        """Test uvw calculation on a timestamp array when the target is a radec target."""
        radec = self.target.radec(self.ts, self.ant1)
        target = katpoint.construct_radec_target(radec.ra, radec.dec)
        u, v, w = target.uvw(self.ant2, np.array([self.ts, self.ts]), self.ant1)
        np.testing.assert_array_almost_equal(u, np.array([self.uvw[0]] * 2), decimal=4)
        np.testing.assert_array_almost_equal(v, np.array([self.uvw[1]] * 2), decimal=4)
        np.testing.assert_array_almost_equal(w, np.array([self.uvw[2]] * 2), decimal=4)

    def test_uvw_antenna_array(self):
        u, v, w = self.target.uvw([self.ant1, self.ant2], self.ts, self.ant1)
        np.testing.assert_array_almost_equal(u, np.array([0, self.uvw[0]]), decimal=5)
        np.testing.assert_array_almost_equal(v, np.array([0, self.uvw[1]]), decimal=5)
        np.testing.assert_array_almost_equal(w, np.array([0, self.uvw[2]]), decimal=5)

    def test_uvw_both_array(self):
        u, v, w = self.target.uvw([self.ant1, self.ant2], [self.ts, self.ts], self.ant1)
        np.testing.assert_array_almost_equal(u, np.array([[0, self.uvw[0]]] * 2), decimal=5)
        np.testing.assert_array_almost_equal(v, np.array([[0, self.uvw[1]]] * 2), decimal=5)
        np.testing.assert_array_almost_equal(w, np.array([[0, self.uvw[2]]] * 2), decimal=5)

    def test_uvw_hemispheres(self):
        """Test uvw calculation near the equator.

        The implementation behaves differently depending on the sign of
        declination. This test is to catch sign flip errors.
        """
        target1 = katpoint.construct_radec_target(0.0, -1e-9)
        target2 = katpoint.construct_radec_target(0.0, +1e-9)
        u1, v1, w1 = target1.uvw(self.ant2, self.ts, self.ant1)
        u2, v2, w2 = target2.uvw(self.ant2, self.ts, self.ant1)
        np.testing.assert_almost_equal(u1, u2, decimal=3)
        np.testing.assert_almost_equal(v1, v2, decimal=3)
        np.testing.assert_almost_equal(w1, w2, decimal=3)

    def test_lmn(self):
        """Test lmn calculation."""
        # For angles less than pi/2, it matches SIN projection
        pointing = katpoint.construct_radec_target('11:00:00.0', '-75:00:00.0')
        target = katpoint.construct_radec_target('16:00:00.0', '-65:00:00.0')
        radec = target.radec(timestamp=self.ts, antenna=self.ant1)
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

    def test_separation(self):
        """Test separation calculation."""
        sun = katpoint.Target('Sun, special')
        azel_sun = sun.azel(self.ts, self.ant1)
        azel = katpoint.construct_azel_target(azel_sun.az, azel_sun.alt)
        sep = sun.separation(azel, self.ts, self.ant1)
        np.testing.assert_almost_equal(sep.rad, 0.0)
        sep = azel.separation(sun, self.ts, self.ant1)
        np.testing.assert_almost_equal(sep.rad, 0.0)
        azel2 = katpoint.construct_azel_target(azel_sun.az,
                                               azel_sun.alt + Angle(0.01, unit=u.rad))
        sep = azel.separation(azel2, self.ts, self.ant1)
        np.testing.assert_almost_equal(sep.rad, 0.01, decimal=7)

    def test_projection(self):
        """Test projection."""
        az, el = katpoint.deg2rad(50.0), katpoint.deg2rad(80.0)
        x, y = self.target.sphere_to_plane(az, el, self.ts, self.ant1)
        re_az, re_el = self.target.plane_to_sphere(x, y, self.ts, self.ant1)
        np.testing.assert_almost_equal(re_az, az, decimal=12)
        np.testing.assert_almost_equal(re_el, el, decimal=12)
