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

"""Enhancements to PyEphem."""

# --------------------------------------------------------------------------------------------------
# --- Helper functions
# --------------------------------------------------------------------------------------------------


def is_iterable(x):
    """Checks if object is iterable (but not a string or 0-dimensional array)."""
    return hasattr(x, '__iter__') and not isinstance(x, str) and \
        not (getattr(x, 'shape', None) == ())
