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

"""Delay model and correction.

This implements the basic delay model used to calculate the delay
contribution from each antenna, as well as a class that performs
delay correction for a correlator.
"""

import logging
import json

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import Angle

from .model import Parameter, Model
from .conversion import azel_to_enu, ecef_to_enu
from .target import construct_radec_target
from .timestamp import Timestamp


# Speed of EM wave in fixed path (typically due to cables / clock distribution).
# This number is not critical - only meant to convert delays to "nice" lengths.
# Typical factors are: fibre = 0.7, coax = 0.84.
LIGHTSPEED = const.c.to_value(u.m / u.s)
FIXEDSPEED = 0.7 * LIGHTSPEED

logger = logging.getLogger(__name__)


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
        params.append(Parameter('POS_E', 'm', 'antenna position: offset East of reference position'))
        params.append(Parameter('POS_N', 'm', 'antenna position: offset North of reference position'))
        params.append(Parameter('POS_U', 'm', 'antenna position: offset above reference position'))
        params.append(Parameter('FIX_H', 'm', 'fixed additional path length for H feed due to electronics / cables'))
        params.append(Parameter('FIX_V', 'm', 'fixed additional path length for V feed due to electronics / cables'))
        params.append(Parameter('NIAO', 'm', 'non-intersecting axis offset - distance between az and el axes'))
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


class DelayCorrection:
    """Calculate delay corrections for a set of correlator inputs / antennas.

    This uses delay models from multiple antennas connected to a correlator to
    produce delay and phase corrections for a given target and timestamp, for
    all correlator inputs at once. The delay corrections are guaranteed to be
    strictly positive. Each antenna is assumed to have two polarisations (H
    and V), resulting in two correlator inputs per antenna.

    Parameters
    ----------
    ants : sequence of *M* :class:`Antenna` objects or str
        Sequence of antennas forming an array and connected to correlator;
        alternatively, a description string representing the entire object
    ref_ant : :class:`Antenna`, optional
        Reference antenna for the array (defaults to first antenna in `ants`)
    sky_centre_freq : :class:`~astropy.units.Quantity`, optional
        RF centre frequency that serves as reference for fringe phase
    extra_delay : :class:`~astropy.units.Quantity`, optional
        Additional delay added to all inputs to ensure strictly positive delay
        corrections (automatically calculated by default)

    Attributes
    ----------
    ant_models : dict mapping str to :class:`DelayModel`
        Dict mapping antenna name to corresponding delay model
    inputs : list of str, length *2M*
        List of correlator input labels corresponding to output of :meth:`delays`

    Raises
    ------
    ValueError
        If description string is invalid
    """

    @u.quantity_input
    def __init__(self, ants, ref_ant=None, sky_centre_freq: u.Hz = 0.0 * u.Hz,
                 extra_delay: u.s = None):
        # Unpack JSON-encoded description string
        if isinstance(ants, str):
            try:
                descr = json.loads(ants)
            except ValueError as err:
                raise ValueError("Trying to construct DelayCorrection with an "
                                 f"invalid description string {ants!r}") from err
            ref_ant_str = descr['ref_ant']
            # Antenna needs DelayModel which also lives in this module...
            # This is messy but avoids a circular dependency and having to
            # split this file into two small bits.
            from .antenna import Antenna
            ref_ant = Antenna(ref_ant_str)
            sky_centre_freq = descr['sky_centre_freq'] * u.Hz
            extra_delay = descr['extra_delay'] * u.s
            ant_models = {}
            for ant_name, ant_model_str in descr['ant_models'].items():
                ant_model = DelayModel()
                ant_model.fromstring(ant_model_str)
                ant_models[ant_name] = ant_model
        else:
            # `ants` is a sequence of Antennas - verify and extract delay models
            if ref_ant is None:
                ref_ant = ants[0]
            ant_models = {}
            for ant in ants:
                model = DelayModel(ant.delay_model)
                # If reference positions agree, keep model to avoid small rounding errors
                if ref_ant.position_wgs84 != ant.ref_position_wgs84:
                    # Remap antenna ENU offset to the common reference position
                    enu = ecef_to_enu(*ref_ant.position_wgs84, *ant.position_ecef)
                    model['POS_E'] = enu[0]
                    model['POS_N'] = enu[1]
                    model['POS_U'] = enu[2]
                ant_models[ant.name] = model

        # Initialise private attributes
        self._params = np.array([ant_models[ant].delay_params
                                 for ant in ant_models])
        # With no antennas, let params still have correct shape
        if not ant_models:
            self._params = np.empty((0, len(DelayModel())))

        # Now calculate and store public attributes
        self.inputs = [ant + pol for ant in ant_models for pol in 'hv']
        self.ant_models = ant_models
        self.ref_ant = ref_ant
        self.sky_centre_freq = sky_centre_freq
        # Add a 1% safety margin to guarantee positive delay corrections
        self.extra_delay = 1.01 * self.max_delay \
            if extra_delay is None else extra_delay

    @property
    @u.quantity_input
    def max_delay(self) -> u.s:
        """The maximum (absolute) delay achievable in the array."""
        # Worst case is wavefront moving along baseline connecting ant to ref
        max_delay_per_ant = np.sqrt((self._params[:, :3] ** 2).sum(axis=1))
        # Pick largest fixed delay
        max_delay_per_ant += self._params[:, 3:5].max(axis=1)
        # Worst case for NIAO is looking at the horizon
        max_delay_per_ant += self._params[:, 5]
        return max(max_delay_per_ant) * u.s if self.ant_models else 0.0 * u.s

    @property
    def description(self):
        """Complete string representation of object that allows reconstruction."""
        descr = {'ref_ant': self.ref_ant.description,
                 'sky_centre_freq': self.sky_centre_freq.to_value(u.Hz),
                 'extra_delay': self.extra_delay.to_value(u.s),
                 'ant_models': {ant: model.description
                                for ant, model in self.ant_models.items()}}
        return json.dumps(descr, sort_keys=True)

    @u.quantity_input
    def delays(self, target, timestamp, offset=None) -> u.s:
        """Calculate delays for all timestamps and inputs for a given target.

        These delays include all geometric effects (also non-intersecting axis
        offsets) and known fixed/cable delays, but not the :attr:`extra_delay`
        needed to make delays strictly positive.

        Parameters
        ----------
        target : :class:`Target`
            Target providing direction for geometric delays
        timestamp : :class:`Timestamp` or equivalent, shape T
            Timestamp(s) when wavefront from target passes reference position
        offset : dict or None, optional
            Keyword arguments for :meth:`Target.plane_to_sphere` to offset
            delay centre relative to target (see method for details)

        Returns
        -------
        delays : :class:`~astropy.units.Quantity`, shape T + (2 * M,)
            Delays for timestamps with shape T and *2M* correlator inputs, with
            ordering on the final axis matching the labels in :attr:`inputs`
        """
        # Ensure a single consistent timestamp in the case of "now"
        if timestamp is None:
            timestamp = Timestamp()
        if offset is None:
            azel = target.azel(timestamp, self.ref_ant)
            az = azel.az.rad
            el = azel.alt.rad
        else:
            coord_system = offset.get('coord_system', 'azel')
            if coord_system == 'radec':
                ra, dec = target.plane_to_sphere(timestamp=timestamp,
                                                 antenna=self.ref_ant, **offset)
                offset_target = construct_radec_target(ra, dec)
                azel = offset_target.azel(timestamp, self.ref_ant)
                az = azel.az.rad
                el = azel.alt.rad
            else:
                az, el = target.plane_to_sphere(timestamp=timestamp,
                                                antenna=self.ref_ant, **offset)
        T = el.shape
        target_dir = np.array(azel_to_enu(az.ravel(), el.ravel()))  # shape (3, prod(T))
        cos_el = np.cos(el.ravel())
        design_mat = np.stack((
            np.vstack((-target_dir, np.ones_like(cos_el), np.zeros_like(cos_el), cos_el)),
            np.vstack((-target_dir, np.zeros_like(cos_el), np.ones_like(cos_el), cos_el)),
        ), axis=1)  # shape (6, 2, prod(T))
        # (A, 6) * (6, 2, prod(T)) => (A, 2, prod(T)) for N inputs, A antennas and N = 2A
        delays = np.tensordot(self._params, design_mat, axes=1)
        # Collapse input dimensions and restore time dimensions, and swap these groups
        delays = np.moveaxis(delays.reshape((-1,) + T), 0, -1)
        return delays * u.s

    def corrections(self, target, timestamp=None, offset=None):
        """Delay and phase corrections for a given target and timestamp(s).

        Calculate delay and phase corrections for the direction towards
        `target` at `timestamp`. Both delay (aka phase slope across frequency)
        and phase (aka phase offset or fringe phase) corrections are provided,
        and their derivatives with respect to time (delay rate and fringe rate,
        respectively). The derivatives allow linear interpolation of delay
        and phase if a sequence of timestamps is provided.

        Parameters
        ----------
        target : :class:`Target`
            Target providing direction for geometric delays
        timestamp : :class:`Timestamp` or equivalent, shape T, optional
            Timestamp(s) when delays are evaluated (default is now)
        offset : dict or None, optional
            Keyword arguments for :meth:`Target.plane_to_sphere` to offset
            delay centre relative to target (see method for details)

        Returns
        -------
        delays : dict mapping str to :class:`~astropy.units.Quantity`, shape T
            Delay correction per correlator input name and timestamp
        phases : dict mapping str to :class:`~astropy.coordinates.Angle`, shape T
            Phase correction per correlator input name and timestamp
        delay_rates : dict mapping str to :class:`~astropy.units.Quantity`
            Delay rate correction per correlator input name and timestamp. The
            quantity shape is like T, except the first dimension is 1 smaller.
            The quantity will be an empty array if there are fewer than 2 times.
        fringe_rates : dict mapping str to :class:`~astropy.units.Quantity`
            Fringe rate correction per correlator input name and timestamp. The
            quantity shape is like T, except the first dimension is 1 smaller.
            The quantity will be an empty array if there are fewer than 2 times.
        """
        time = Timestamp(timestamp).time
        T = time.shape
        # Ensure that times are at least 1-D (and delays 2-D) so that we can calculate deltas
        time = np.atleast_1d(time)
        delays = self.delays(target, time, offset)
        delay_corrections = self.extra_delay - delays
        # The phase term is (-2 pi freq delay) so the correction is (+2 pi freq delay)
        turns = (self.sky_centre_freq * delays).decompose()
        phase_corrections = Angle(2. * np.pi * u.rad) * turns
        delta_time = (time[1:] - time[:-1])[:, np.newaxis].to(u.s)
        delta_delay_corrections = delay_corrections[1:] - delay_corrections[:-1]
        delay_rate_corrections = delta_delay_corrections / delta_time
        delta_phase_corrections = phase_corrections[1:] - phase_corrections[:-1]
        fringe_rate_corrections = delta_phase_corrections / delta_time
        return (
            # Restore time dimensions and swap input dimension to front to get zipped into dict
            dict(zip(self.inputs, np.moveaxis(delay_corrections.reshape(T + (-1,)), -1, 0))),
            dict(zip(self.inputs, np.moveaxis(phase_corrections.reshape(T + (-1,)), -1, 0))),
            dict(zip(self.inputs, np.moveaxis(delay_rate_corrections, -1, 0))),
            dict(zip(self.inputs, np.moveaxis(fringe_rate_corrections, -1, 0))),
        )
