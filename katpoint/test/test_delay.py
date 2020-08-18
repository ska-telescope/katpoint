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
from astropy import units as u
from astropy.coordinates import Angle

import katpoint


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
        self.delays = katpoint.DelayCorrection([self.ant2, self.ant3], self.ant1, 1.285e9)

    def test_construction(self):
        """Test construction of DelayCorrection object."""
        descr = self.delays.description
        delays2 = katpoint.DelayCorrection(descr)
        delays_dict = json.loads(descr)
        delays2_dict = json.loads(delays2.description)
        assert delays2_dict == delays_dict, 'Objects created through description strings differ'
        with pytest.raises(ValueError):
            katpoint.DelayCorrection([self.ant1, self.ant2], self.ant3)
        with pytest.raises(ValueError):
            katpoint.DelayCorrection([self.ant1, self.ant2])
        with pytest.raises(ValueError):
            katpoint.DelayCorrection('')
        delays3 = katpoint.DelayCorrection([], self.ant1)
        assert delays3._params.shape == (0, len(katpoint.DelayModel())), (
            "Delay correction with no antennas should fail gracefully")

    def test_correction(self):
        """Test delay correction."""
        extra_delay = self.delays.extra_delay
        delay0, phase0 = self.delays.corrections(self.target1, self.ts)
        delay1, phase1 = self.delays.corrections(self.target1, self.ts, self.ts + 1.0)
        # This target is special - direction perpendicular to baseline (and stationary)
        assert delay0['A2h'] == extra_delay, 'Delay for ant2h should be zero'
        assert delay0['A2v'] == extra_delay, 'Delay for ant2v should be zero'
        assert delay1['A2h'][0] == extra_delay, 'Delay for ant2h should be zero'
        assert delay1['A2v'][0] == extra_delay, 'Delay for ant2v should be zero'
        assert delay1['A2h'][1] == 0.0, 'Delay rate for ant2h should be zero'
        assert delay1['A2v'][1] == 0.0, 'Delay rate for ant2v should be zero'
        # Compare to target geometric delay calculations
        delay0, phase0 = self.delays.corrections(self.target2, self.ts)
        delay1, phase1 = self.delays.corrections(self.target2, self.ts - 0.5, self.ts + 0.5)
        tgt_delay, tgt_delay_rate = self.target2.geometric_delay(self.ant2, self.ts, self.ant1)
        np.testing.assert_almost_equal(delay0['A2h'], extra_delay - tgt_delay, decimal=15)
        np.testing.assert_almost_equal(delay1['A2h'][1], -tgt_delay_rate, decimal=13)
        # Test vector version
        delay2, phase2 = self.delays.corrections(self.target2, (self.ts - 0.5, self.ts + 0.5))
        np.testing.assert_equal(delay2['A2h'][0], delay1['A2h'])
        np.testing.assert_equal(phase2['A2h'][0], phase1['A2h'])

    def test_delay_cache(self):
        """Test delay correction cache limit."""
        max_size = katpoint.DelayCorrection.CACHE_SIZE
        for n in range(max_size + 10):
            delay0, phase0 = self.delays.corrections(self.target1, self.ts + n)
        assert len(self.delays._cache) == max_size, 'Delay cache grew past limit'

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
        delay0, phase0 = self.delays.corrections(target3, self.ts, offset=offset)
        delay1, phase1 = self.delays.corrections(target3, self.ts, self.ts + 1.0, offset)
        # Conspire to return to special target1
        assert delay0['A2h'] == extra_delay, 'Delay for ant2h should be zero'
        assert delay0['A2v'] == extra_delay, 'Delay for ant2v should be zero'
        assert delay1['A2h'][0] == extra_delay, 'Delay for ant2h should be zero'
        assert delay1['A2v'][0] == extra_delay, 'Delay for ant2v should be zero'
        assert delay1['A2h'][1] == 0.0, 'Delay rate for ant2h should be zero'
        assert delay1['A2v'][1] == 0.0, 'Delay rate for ant2v should be zero'
        # Now try (ra, dec) coordinate system
        radec = self.target1.radec(self.ts, self.ant1)
        offset = dict(projection_type='ARC', coord_system='radec')
        target4 = katpoint.construct_radec_target(radec.ra - Angle(1.0, unit=u.deg),
                                                  radec.dec - Angle(1.0, unit=u.deg))
        x, y = target4.sphere_to_plane(radec.ra.rad, radec.dec.rad, self.ts, self.ant1, **offset)
        offset['x'] = x
        offset['y'] = y
        extra_delay = self.delays.extra_delay
        delay0, phase0 = self.delays.corrections(target4, self.ts, offset=offset)
        delay1, phase1 = self.delays.corrections(target4, self.ts, self.ts + 1.0, offset)
        # Conspire to return to special target1
        # np.testing.assert_almost_equal(delay0['A2h'], extra_delay, decimal=15)
        # np.testing.assert_almost_equal(delay0['A2v'], extra_delay, decimal=15)
        # np.testing.assert_almost_equal(delay1['A2h'][0], extra_delay, decimal=15)
        # np.testing.assert_almost_equal(delay1['A2v'][0], extra_delay, decimal=15)
        # np.testing.assert_almost_equal(delay1['A2h'][1], 0.0, decimal=15)
        # np.testing.assert_almost_equal(delay1['A2v'][1], 0.0, decimal=15)
