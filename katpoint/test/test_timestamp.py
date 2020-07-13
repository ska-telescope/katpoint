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

"""Tests for the timestamp module."""

import pytest
from astropy.time import Time

import katpoint


class TestTimestamp:
    """Test timestamp creation and conversion."""

    def setup(self):
        self.valid_timestamps = [(1248186982.3980861, '2009-07-21 14:36:22.398'),
                                 (Time('2009-07-21 02:52:12.34'), '2009-07-21 02:52:12.340'),
                                 (0, '1970-01-01 00:00:00'),
                                 (-10, '1969-12-31 23:59:50'),
                                 ('2009-07-21 02:52:12.034', '2009-07-21 02:52:12.034'),
                                 ('2009-07-21 02:52:12.000', '2009-07-21 02:52:12'),
                                 ('2009-07-21 02:52:12', '2009-07-21 02:52:12'),
                                 ('2009-07-21 02:52', '2009-07-21 02:52:00'),
                                 ('2009-07-21', '2009-07-21 00:00:00'),
                                 ('2009/07/21 02:52:12.034', '2009-07-21 02:52:12.034'),
                                 ('2009/07/21 02:52:12.000', '2009-07-21 02:52:12'),
                                 ('2009/07/21 02:52:12', '2009-07-21 02:52:12'),
                                 ('2009/07/21 02:52', '2009-07-21 02:52:00'),
                                 ('2009/07/21', '2009-07-21 00:00:00'),
                                 ('2019-07-21 02:52:12', '2019-07-21 02:52:12')]
        self.invalid_timestamps = ['gielie', '03 Mar 2003']
        self.overflow_timestamps = ['2049-07-21 02:52:12']

    def test_construct_timestamp(self):
        """Test construction of timestamps."""
        for v, s in self.valid_timestamps:
            t = katpoint.Timestamp(v)
            assert str(t) == s, (
                "Timestamp string ('%s') differs from expected one ('%s')"
                % (str(t), s))
        for v in self.invalid_timestamps:
            with pytest.raises(ValueError):
                katpoint.Timestamp(v)
#        for v in self.overflow_timestamps:
#            with pytest.raises(OverflowError):
#               katpoint.Timestamp(v)

    def test_numerical_timestamp(self):
        """Test numerical properties of timestamps."""
        t = katpoint.Timestamp(self.valid_timestamps[0][0])
        assert t == t + 0.0
        assert t != t + 1.0
        assert t > t - 1.0
        assert t < t + 1.0
        assert t == eval('katpoint.' + repr(t))
        assert float(t) == self.valid_timestamps[0][0]
        t = katpoint.Timestamp(self.valid_timestamps[1][0])
        assert t.time == self.valid_timestamps[1][0]
        try:
            assert hash(t) == hash(t + 0.0), 'Timestamp hashes not equal'
        except TypeError:
            pytest.fail('Timestamp object not hashable')

    def test_operators(self):
        """Test operators defined for timestamps."""
        T = katpoint.Timestamp(self.valid_timestamps[0][0])
        S = T.secs
        # Logical operators, float treated as absolute time
        assert T == S
        assert T < S + 1
        assert T > S - 1
        # Arithmetic operators, float treated as interval
        assert isinstance(T - S, katpoint.Timestamp)
        assert isinstance(S - T, float)
        assert isinstance(T - T, float)
