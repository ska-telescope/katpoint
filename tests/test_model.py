################################################################################
# Copyright (c) 2014,2017-2018,2020-2021,2023, National Research Foundation (SARAO)
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

from io import StringIO

import pytest

import katpoint


def params():
    """Generate fresh set of parameters (otherwise models share the same ones)."""
    return [
        katpoint.Parameter("POS_E", "m", "East", value=10.0),
        katpoint.Parameter("POS_N", "m", "North", value=-9.0),
        katpoint.Parameter("POS_U", "m", "Up", value=3.0),
        katpoint.Parameter("NIAO", "m", "non-inter", value=0.88),
        katpoint.Parameter("CAB_H", "", "horizontal", value=20.2),
        katpoint.Parameter("CAB_V", "deg", "vertical", value=20.3),
    ]


def test_construct_save_load():
    """Test construction / save / load of generic model."""
    m = katpoint.Model(params())
    m.header["date"] = "2014-01-15"
    # Exercise all string representations for coverage purposes
    print(f"{m!r} {m} {m.params['POS_E']!r}")
    # An empty file should lead to a BadModelFile exception
    cfg_file = StringIO()
    with pytest.raises(katpoint.BadModelFile):
        m.fromfile(cfg_file)
    m.tofile(cfg_file)
    cfg_str = cfg_file.getvalue()
    cfg_file.close()
    # Load the saved config file
    cfg_file = StringIO(cfg_str)
    m2 = katpoint.Model(params())
    m2.fromfile(cfg_file)
    assert m == m2, "Saving model to file and loading it again failed"
    cfg_file = StringIO(cfg_str)
    m2.set(cfg_file)
    assert m == m2, "Saving model to file and loading it again failed"
    # Build model from description string
    m3 = katpoint.Model(params())
    m3.fromstring(m.description)
    assert m == m3, "Saving model to string and loading it again failed"
    m3.set(m.description)
    assert m == m3, "Saving model to string and loading it again failed"
    # Build model from sequence of floats
    m4 = katpoint.Model(params())
    m4.fromlist(m.values())
    assert m == m4, "Saving model to list and loading it again failed"
    m4.set(m.values())
    assert m == m4, "Saving model to list and loading it again failed"
    # Empty model
    cfg_file = StringIO("[header]\n[params]\n")
    m5 = katpoint.Model(params())
    m5.fromfile(cfg_file)
    print(m5)
    assert m != m5, "Model should not be equal to an empty one"
    m6 = katpoint.Model(params())
    m6.set()
    assert m6 == m5, "Setting empty model failed"
    m7 = katpoint.Model(params())
    m7.set(m)
    assert m == m7, "Construction from model object failed"

    class OtherModel(katpoint.Model):
        """A different Model type to check that one type cannot construct another."""

    m8 = OtherModel(params())
    with pytest.raises(katpoint.BadModelFile):
        m8.set(m)
    try:
        assert hash(m) == hash(m4), "Model hashes not equal"
    except TypeError:
        pytest.fail("Model object not hashable")


def test_dict_interface():
    """Test dict interface of generic model."""
    parameters = params()
    names = [p.name for p in parameters]
    values = [p.value for p in parameters]
    m = katpoint.Model(parameters)
    assert len(m) == 6, "Unexpected model length"
    assert list(m.keys()) == names, "Parameter names do not match"
    assert list(m.values()) == values, "Parameter values do not match"
    m["NIAO"] = 6789.0
    assert m["NIAO"] == 6789.0, "Parameter setting via dict interface failed"
