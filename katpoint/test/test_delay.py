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

"""Tests for the model module."""

import json
from io import StringIO

import pytest
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle

import katpoint

try:
    from almacalc.highlevel import calc
except ImportError:
    HAS_ALMACALC = False
else:
    HAS_ALMACALC = True


def test_construct_save_load():
    """Test construction / save / load of delay model."""
    m = katpoint.DelayModel('1.0, -2.0, -3.0, 4.123, 5.0, 6.0')
    m.header['date'] = '2014-01-15'
    # An empty file should lead to a BadModelFile exception
    cfg_file = StringIO()
    with pytest.raises(katpoint.BadModelFile):
        m.fromfile(cfg_file)
    m.tofile(cfg_file)
    cfg_str = cfg_file.getvalue()
    cfg_file.close()
    # Load the saved config file
    cfg_file = StringIO(cfg_str)
    m2 = katpoint.DelayModel()
    m2.fromfile(cfg_file)
    assert m == m2, 'Saving delay model to file and loading it again failed'
    params = m.delay_params
    m3 = katpoint.DelayModel()
    m3.fromdelays(params)
    assert m == m3, 'Converting delay model to delay parameters and loading it again failed'
    try:
        assert hash(m) == hash(m3), 'Delay model hashes not equal'
    except TypeError:
        pytest.fail('DelayModel object not hashable')


class TestDelayCorrection:
    """Test correlator delay corrections."""

    def setup(self):
        self.target1 = katpoint.construct_azel_target('45:00:00.0', '75:00:00.0')
        self.target2 = katpoint.Target('Sun, special')
        self.ant1 = katpoint.Antenna('A1, -31.0, 18.0, 0.0, 12.0, 0.0 0.0 0.0')
        self.ant2 = katpoint.Antenna('A2, -31.0, 18.0, 0.0, 12.0, 10.0 -10.0 0.0')
        self.ant3 = katpoint.Antenna('A3, -31.0, 18.0, 0.0, 12.0, 5.0 10.0 3.0')
        self.ts = katpoint.Timestamp('2013-08-14 08:25')
        self.delays = katpoint.DelayCorrection([self.ant2, self.ant3], self.ant1,
                                               1.285 * u.GHz)

    def test_construction(self):
        """Test construction of DelayCorrection object."""
        descr = self.delays.description
        assert self.delays.inputs == ['A2h', 'A2v', 'A3h', 'A3v']
        delays2 = katpoint.DelayCorrection(descr)
        delays_dict = json.loads(descr)
        delays2_dict = json.loads(delays2.description)
        assert delays2_dict == delays_dict, 'Objects created through description strings differ'
        with pytest.raises(ValueError):
            katpoint.DelayCorrection('')
        delays3 = katpoint.DelayCorrection([], self.ant1)
        d = delays3.delays(self.target1, self.ts + np.arange(3))
        assert d.shape == (3, 0), "Delay correction with no antennas should fail gracefully"
        # Check construction with different antenna reference positions
        delays4 = katpoint.DelayCorrection([self.ant1, self.ant2], self.ant3)
        ant1_vs_ant3 = np.array(delays4.ant_models['A1'].values())
        ant3_vs_ant1 = np.array(self.delays.ant_models['A3'].values())
        assert np.allclose(ant3_vs_ant1, -ant1_vs_ant3, rtol=0, atol=2e-5)
        delays5 = katpoint.DelayCorrection([self.ant1, self.ant2])
        assert delays5.ref_ant == self.ant1

    def test_correction(self):
        """Test delay correction."""
        extra_delay = self.delays.extra_delay
        delay0, phase0, drate0, frate0 = self.delays.corrections(self.target1, self.ts)
        delay1, phase1, drate1, frate1 = self.delays.corrections(self.target1,
                                                                 [self.ts, self.ts + 1.0])
        # First check dimensions for time dimension T0 = () and T1 = (2,), respectively
        assert np.shape(delay0['A2h']) == np.shape(phase0['A2h']) == ()
        assert np.shape(drate0['A2h']) == np.shape(frate0['A2h']) == (0,)
        assert np.shape(delay1['A2h']) == np.shape(phase1['A2h']) == (2,)
        assert np.shape(drate1['A2h']) == np.shape(frate1['A2h']) == (1,)
        # This target is special - direction perpendicular to baseline (and stationary)
        assert delay0['A2h'] == delay0['A2v'] == extra_delay
        assert drate1['A2h'] == drate1['A2v'] == [0.0]
        assert frate1['A2h'] == frate1['A2v'] == [0.0]
        np.testing.assert_array_equal(delay1['A2h'], extra_delay.repeat(2))
        np.testing.assert_array_equal(delay1['A2v'], extra_delay.repeat(2))
        np.testing.assert_array_equal(drate1['A2h'], np.array([0.0]))
        np.testing.assert_array_equal(drate1['A2v'], np.array([0.0]))
        np.testing.assert_array_equal(frate1['A2h'], np.array([0.0]) * u.rad / u.s)
        np.testing.assert_array_equal(frate1['A2v'], np.array([0.0]) * u.rad / u.s)
        # Compare to target geometric delay calculations
        delay0, _, _, _ = self.delays.corrections(self.target2, self.ts)
        _, _, drate1, _ = self.delays.corrections(self.target2, (self.ts - 0.5, self.ts + 0.5))
        tgt_delay, tgt_delay_rate = self.target2.geometric_delay(self.ant2, self.ts, self.ant1)
        assert np.allclose(delay0['A2h'], extra_delay - tgt_delay * u.s, atol=0, rtol=1e-15)
        assert np.allclose(drate1['A2h'][0], -tgt_delay_rate * u.s / u.s, atol=0, rtol=1e-11)

    def test_offset(self):
        """Test target offset."""
        azel = self.target1.azel(self.ts, self.ant1)
        offset = dict(projection_type='SIN')
        target3 = katpoint.construct_azel_target(azel.az - Angle(1.0, unit=u.deg),
                                                 azel.alt - Angle(1.0, unit=u.deg))
        x, y = target3.sphere_to_plane(azel.az.rad, azel.alt.rad, self.ts, self.ant1, **offset)
        offset['x'] = x
        offset['y'] = y
        extra_delay = self.delays.extra_delay
        delay0, _, _, _ = self.delays.corrections(target3, self.ts, offset=offset)
        delay1, _, drate1, _ = self.delays.corrections(target3, (self.ts, self.ts + 1.0), offset)
        # Conspire to return to special target1
        assert delay0['A2h'] == extra_delay, 'Delay for ant2h should be zero'
        assert delay0['A2v'] == extra_delay, 'Delay for ant2v should be zero'
        np.testing.assert_array_equal(delay1['A2h'], extra_delay.repeat(2))
        np.testing.assert_array_equal(delay1['A2v'], extra_delay.repeat(2))
        np.testing.assert_array_equal(drate1['A2h'], np.array([0.0]))
        np.testing.assert_array_equal(drate1['A2v'], np.array([0.0]))
        # Now try (ra, dec) coordinate system
        radec = self.target1.radec(self.ts, self.ant1)
        offset = dict(projection_type='ARC', coord_system='radec')
        target4 = katpoint.construct_radec_target(radec.ra - Angle(1.0, unit=u.deg),
                                                  radec.dec - Angle(1.0, unit=u.deg))
        x, y = target4.sphere_to_plane(radec.ra.rad, radec.dec.rad, self.ts, self.ant1, **offset)
        offset['x'] = x
        offset['y'] = y
        extra_delay = self.delays.extra_delay
        delay0, _, _, _ = self.delays.corrections(target4, self.ts, offset=offset)
        delay1, _, drate1, _ = self.delays.corrections(target4, (self.ts, self.ts + 1.0), offset)
        # Conspire to return to special target1
        assert np.allclose(delay0['A2h'], extra_delay, atol=0, rtol=1e-12)
        assert np.allclose(delay0['A2v'], extra_delay, atol=0, rtol=1e-12)
        assert np.allclose(delay1['A2h'][0], extra_delay, atol=0, rtol=1e-12)
        assert np.allclose(delay1['A2v'][0], extra_delay, atol=0, rtol=1e-12)
        assert np.allclose(drate1['A2h'], [0.0], atol=5e-12)
        assert np.allclose(drate1['A2v'], [0.0], atol=5e-12)


TARGET = katpoint.Target('J1939-6342, radec, 19:39:25.03, -63:42:45.6')
DELAY_MODEL = {'ref_ant': 'array, -30:42:39.8, 21:26:38, 1086.6, 0',
               'extra_delay': 0.0, 'sky_centre_freq': 1284000000.0}


@pytest.mark.skipif(not HAS_ALMACALC, reason="almacalc is not installed")
@pytest.mark.parametrize(
    "times,ant_models,min_diff,max_diff",
    [
        (1605646800.0 + np.linspace(0, 86400, 9),
         {'m063': '-3419.5845 -1840.48 16.3825'}, 14 * u.ps, 16 * u.ps),
        (1571219913.0 + np.arange(0, 54000, 6000),
         {'m048': '-2805.653 2686.863 -9.7545',
          'm058': '2805.764 2686.873 -3.6595',
          's0121': '-3545.28803 -10207.44399 -9.18584'}, 12 * u.ps, 16 * u.ps),
    ]
)
def test_against_calc(times, ant_models, min_diff, max_diff):
    times = katpoint.Timestamp(times)
    model = dict(ant_models=ant_models, **DELAY_MODEL)
    dc = katpoint.DelayCorrection(json.dumps(model))
    delay = dc.delays(TARGET, times)[:, ::2]
    ref_location = katpoint.Antenna(model['ref_ant']).location
    locations = np.stack([katpoint.Antenna(f"{model['ref_ant']}, {dm}").location
                          for dm in model['ant_models'].values()])
    expected_delay = calc(locations, TARGET.body.coord, times.time, ref_location)
    abs_diff = np.abs(delay - expected_delay)
    assert np.all(abs_diff == np.clip(abs_diff, min_diff, max_diff))
