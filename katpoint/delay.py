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
from astropy.time import Time

from .model import Parameter, Model
from .conversion import azel_to_enu
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

    For now, the reference antenna position must match the reference positions
    of each antenna in the array, so that the ENU offset in each antenna's
    delay model directly represent the baseline between that antenna and the
    reference antenna. This should be fine as this is the standard case, but
    may cause problems for e.g. VLBI with a geocentric reference antenna.

    Parameters
    ----------
    ants : sequence of *M* :class:`Antenna` objects or string
        Sequence of antennas forming an array and connected to correlator;
        alternatively, a description string representing the entire object
    ref_ant : :class:`Antenna` object or None, optional
        Reference antenna for the array (only optional if `ants` is a string)
    sky_centre_freq : float, optional
        RF centre frequency that serves as reference for fringe phase
    extra_delay : None or float, optional
        Additional delay, in seconds, added to all inputs to ensure strictly
        positive delay corrections (automatically calculated if None)

    Attributes
    ----------
    ant_models : dict mapping string to :class:`DelayModel` object
        Dict mapping antenna name to corresponding delay model
    inputs : list of strings
        List of correlator input labels corresponding to output of :meth:`delays`

    Raises
    ------
    ValueError
        If all antennas do not share the same reference position as `ref_ant`
        or `ref_ant` was not specified, or description string is invalid
    """

    def __init__(self, ants, ref_ant=None, sky_centre_freq=0.0, extra_delay=None):
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
            sky_centre_freq = descr['sky_centre_freq']
            extra_delay = descr['extra_delay']
            ant_models = {}
            for ant_name, ant_model_str in descr['ant_models'].items():
                ant_model = DelayModel()
                ant_model.fromstring(ant_model_str)
                ant_models[ant_name] = ant_model
        else:
            # `ants` is a sequence of Antennas - verify and extract delay models
            if ref_ant is None:
                raise ValueError('No reference antenna provided')
            # Tolerances translate to micrometre differences (assume float64)
            if any([not np.allclose(ant.ref_position_wgs84,
                                    ref_ant.position_wgs84, rtol=0., atol=1e-14)
                    for ant in list(ants) + [ref_ant]]):
                msg = "Antennas '%s' do not all share the same reference " \
                      "position of the reference antenna %r" % \
                      ("', '".join(ant.description for ant in ants),
                       ref_ant.description)
                raise ValueError(msg)
            ant_models = {ant.name: ant.delay_model for ant in ants}

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
    def max_delay(self):
        """The maximum (absolute) delay achievable in the array, in seconds."""
        # Worst case is wavefront moving along baseline connecting ant to ref
        max_delay_per_ant = np.sqrt((self._params[:, :3] ** 2).sum(axis=1))
        # Pick largest fixed delay
        max_delay_per_ant += self._params[:, 3:5].max(axis=1)
        # Worst case for NIAO is looking at the horizon
        max_delay_per_ant += self._params[:, 5]
        return max(max_delay_per_ant) if self.ant_models else 0.0

    @property
    def description(self):
        """Complete string representation of object that allows reconstruction."""
        descr = {'ref_ant': self.ref_ant.description,
                 'sky_centre_freq': self.sky_centre_freq,
                 'extra_delay': self.extra_delay,
                 'ant_models': {ant: model.description
                                for ant, model in self.ant_models.items()}}
        return json.dumps(descr, sort_keys=True)

    def delays(self, target, timestamp, offset=None):
        """Calculate delays for all inputs / antennas for a given target.

        Parameters
        ----------
        target : :class:`Target` object
            Target providing direction for geometric delays
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equivalent
            Timestamp when wavefront from target passes reference position
        offset : dict or None, optional
            Keyword arguments for :meth:`Target.plane_to_sphere` to offset
            delay centre relative to target (see method for details)

        Returns
        -------
        delays : sequence of *2M* floats
            Delays (one per correlator input) in seconds
        """
        if not offset:
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
        targetdir = np.array(azel_to_enu(az, el))
        cos_el = np.cos(el)
        design_mat = np.array([np.r_[-targetdir, 1.0, 0.0, cos_el],
                               np.r_[-targetdir, 0.0, 1.0, cos_el]])
        return np.dot(self._params, design_mat.T).ravel()

    def corrections(self, target, timestamp=None, next_timestamp=None,
                    offset=None):
        """Delay and phase corrections for a given target and timestamp(s).

        Calculate delay and phase corrections for the direction towards
        *target* at *timestamp*. If the timestamp of the next delay
        calculation is provided, it is used to calculate a delay rate that can
        be used for linear interpolation in the period up to the next update.
        This process is repeated if a sequence of timestamps is given. Both
        delay (aka phase slope) and phase (aka phase offset or fringe phase)
        corrections are provided, and optionally their derivatives with
        respect to time (delay rate and fringe rate, respectively).

        Parameters
        ----------
        target : :class:`Target` object
            Target providing direction for geometric delays
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equivalent, optional
            Timestamp(s) when delays are evaluated (default is now). If an array
            of timestamps is given (in which case, it must contain at least two
            elements), the corrections will include slopes to be used for linear
            interpolation between the times.
        next_timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equivalent, optional
            Timestamp when next delay will be evaluated, used to determine
            a slope for linear interpolation (default is no slope). This is
            ignored if *timestamp* is a sequence.
        offset : dict or None, optional
            Keyword arguments for :meth:`Target.plane_to_sphere` to offset
            delay centre relative to target (see method for details)

        Returns
        -------
        delays : dict mapping string to float or array of floats
            Dict mapping correlator input name to delay correction,
            which consists of a delay value (in seconds) and optionally
            a delay rate value (in seconds per second). If a sequence
            of *T* timestamps are provided, each input maps to an array
            of shape (*T*, 2).
        phases : dict mapping string to float or array of floats
            Dict mapping correlator input name to phase correction, which
            consists of a fringe phase value (in radians) and optionally a
            fringe rate value (in radians per second). If a sequence of *T*
            timestamps are provided, each input maps to an array of shape
            (*T*, 2).
        """
        time = Timestamp(timestamp).time
        if time.shape == ():
            delays = self.delays(target, time, offset)
            next_time = None if next_timestamp is None else Timestamp(next_timestamp).time
        else:
            # Append one more timestamp to get a slope for the last timestamp
            last_step = time[-1] - time[-2]
            all_times = np.r_[time, [time[-1] + last_step]]
            next_time = Time(all_times[1:])
            all_delays = np.array([self.delays(target, t, offset) for t in all_times])
            delays = all_delays[:-1].T
            next_delays = all_delays[1:].T

        def phase(t0):
            """The phase associated with delay t0 at the centre frequency."""
            return - 2.0 * np.pi * self.sky_centre_freq * t0
        delay_corrections = self.extra_delay - delays
        phase_corrections = - phase(delays)
        if next_time is None:
            return (dict(zip(self.inputs, delay_corrections)),
                    dict(zip(self.inputs, phase_corrections)))
        step = (next_time - time).sec
        # We still have to get next_delays in the single timestamp case
        if next_time.shape == ():
            next_delays = self.delays(target, next_time, offset)
        next_delay_corrections = self.extra_delay - next_delays
        next_phase_corrections = - phase(next_delays)
        delay_slopes = (next_delay_corrections - delay_corrections) / step
        phase_slopes = (next_phase_corrections - phase_corrections) / step
        # This construction works for both the scalar and vector cases.
        # The squeeze() gets rid of an extra singleton in the scalar case.
        # It is safe to squeeze as the other two dimensions involved will
        # never be singletons (number of inputs >= 2 even for 1 antenna, and
        # number of polynomial terms is 2 by design).
        delay_polys = np.dstack((delay_corrections, delay_slopes)).squeeze()
        phase_polys = np.dstack((phase_corrections, phase_slopes)).squeeze()
        return (dict(zip(self.inputs, delay_polys)),
                dict(zip(self.inputs, phase_polys)))
