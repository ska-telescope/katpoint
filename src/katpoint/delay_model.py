################################################################################
# Copyright (c) 2013-2021,2023, National Research Foundation (SARAO)
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

"""Delay model.

This implements the basic delay model used to calculate the delay
contribution from each antenna.
"""

import astropy.constants as const
import astropy.units as u
import numpy as np

from .model import Model, Parameter

# Speed of EM wave in fixed path (typically due to cables / clock distribution).
# This number is not critical - only meant to convert delays to "nice" lengths.
# Typical factors are: fibre = 0.7, coax = 0.84.
LIGHTSPEED = const.c.to_value(u.m / u.s)
FIXEDSPEED = 0.7 * LIGHTSPEED


class DelayModel(Model):
    """Model of the delay contribution from a single antenna.

    This object is purely used as a repository for model parameters, allowing
    easy construction, inspection and saving of the delay model. The actual
    calculations happen in :class:`DelayCorrection`, which is more efficient
    as it handles multiple antenna delays simultaneously.

    Parameters
    ----------
    model : file-like or model object, sequence of floats, or string, optional
        Model specification. If this is a file-like or model object, load the
        model from it. If this is a sequence of floats, accept it directly as
        the model parameters (defaults to sequence of zeroes). If it is a
        string, interpret it as a comma-separated (or whitespace-separated)
        sequence of parameters in their string form (i.e. a description
        string). The default is an empty model.
    """

    def __init__(self, model=None):
        # Instantiate the relevant model parameters and register with base class
        params = []
        params.append(
            Parameter(
                "POS_E", "m", "antenna position: offset East of reference position"
            )
        )
        params.append(
            Parameter(
                "POS_N", "m", "antenna position: offset North of reference position"
            )
        )
        params.append(
            Parameter("POS_U", "m", "antenna position: offset above reference position")
        )
        params.append(
            Parameter(
                "FIX_H",
                "m",
                "fixed additional path length for H feed due to electronics / cables",
            )
        )
        params.append(
            Parameter(
                "FIX_V",
                "m",
                "fixed additional path length for V feed due to electronics / cables",
            )
        )
        params.append(
            Parameter(
                "NIAO",
                "m",
                "non-intersecting axis offset - distance between az and el axes",
            )
        )
        Model.__init__(self, params)
        self.set(model)
        # The EM wave velocity associated with each parameter
        self._speeds = np.array([LIGHTSPEED] * 3 + [FIXEDSPEED] * 2 + [LIGHTSPEED])

    @property
    def delay_params(self):
        """The model parameters converted to delays in seconds."""
        return np.array(self.values()) / self._speeds

    def fromdelays(self, delays):
        """Update model from a sequence of delay parameters.

        Parameters
        ----------
        delays : sequence of floats
            Model parameters in delay form (i.e. in seconds)
        """
        self.fromlist(delays * self._speeds)
