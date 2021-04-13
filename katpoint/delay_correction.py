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

"""Delay correction.

This implements a class that performs delay correction for a correlator.
"""

import json

import numpy as np
import astropy.units as u
from astropy.coordinates import Angle

from .delay_model import DelayModel
from .antenna import Antenna
from .conversion import azel_to_enu, ecef_to_enu
from .target import Target
from .timestamp import Timestamp
from .refraction import TroposphericDelay


NO_TEMPERATURE = -300 * u.deg_C  # used as default parameter, akin to None


class DelayCorrection:
    """Calculate delay corrections for a set of correlator inputs / antennas.

    This uses delay models from multiple antennas connected to a correlator to
    produce delay and phase corrections for a given target and timestamp, for
    all correlator inputs at once. The delay corrections are guaranteed to be
    strictly positive. Each antenna is assumed to have two polarisations (H
    and V), resulting in two correlator inputs per antenna.

    Parameters
    ----------
    ants : sequence of *A* :class:`Antenna` objects or str
        Sequence of antennas forming an array and connected to correlator;
        alternatively, a description string representing the entire object
    ref_ant : :class:`Antenna`, optional
        Reference antenna for the array (defaults to first antenna in `ants`)
    sky_centre_freq : :class:`~astropy.units.Quantity`, optional
        RF centre frequency that serves as reference for fringe phase
    extra_correction : :class:`~astropy.units.Quantity`, optional
        Additional correction added to all inputs to ensure strictly positive
        corrections (automatically calculated by default)
    tropospheric_model : str, optional
        Unique identifier of tropospheric model, or 'None' if no tropospheric
        correction will be done

    Attributes
    ----------
    ant_models : dict mapping str to :class:`DelayModel`
        Dict mapping antenna name to corresponding delay model
    inputs : list of str, length *2A*
        List of correlator input labels corresponding to output of :meth:`delays`
    ant_locations : :class:`~astropy.coordinates.EarthLocation`, shape (A,)
        Locations of *A* antennas, in the same order as `inputs`
    ref_location : :class:`~astropy.coordinates.EarthLocation`
        Location of reference antenna

    Raises
    ------
    ValueError
        If description string is invalid
    """

    @u.quantity_input
    def __init__(self, ants, ref_ant=None, sky_centre_freq: u.Hz = 0.0 * u.Hz,
                 extra_correction: u.s = None, tropospheric_model='None'):
        # Unpack JSON-encoded description string
        if isinstance(ants, str):
            try:
                descr = json.loads(ants)
            except ValueError as err:
                raise ValueError("Trying to construct DelayCorrection with an "
                                 f"invalid description string {ants!r}") from err
            ref_ant = Antenna(descr['ref_ant'])
            sky_centre_freq = descr['sky_centre_freq'] * u.Hz
            try:
                extra_correction = descr['extra_correction'] * u.s
            except KeyError:
                # Also try the older name of this attribute to remain backwards compatible
                try:
                    extra_correction = descr['extra_delay'] * u.s
                except KeyError:
                    raise KeyError("no 'extra_correction' or 'extra_delay'") from None
            tropospheric_model = descr.get('tropospheric_model', 'None')
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

        # Delay model parameters in units of seconds are combined in array of shape (A, 6)
        self._params = np.array([ant_models[ant].delay_params for ant in ant_models]) * u.s
        # With no antennas, let params still have correct shape
        self._params.shape = (-1, len(DelayModel()))
        if tropospheric_model in ('None', None):
            self._tropospheric_delay = None
        else:
            # XXX There should ideally be a TroposphericDelay object per actual antenna
            self._tropospheric_delay = TroposphericDelay(ref_ant.location, tropospheric_model)

        # Now calculate and store public attributes
        self.ant_models = ant_models
        self.ref_ant = ref_ant
        self.sky_centre_freq = sky_centre_freq
        # Add a 1% safety margin to guarantee positive delay corrections
        self.extra_correction = 1.01 * self.max_delay \
            if extra_correction is None else extra_correction
        self.inputs = [ant + pol for ant in ant_models for pol in 'hv']
        self._locations = np.stack([Antenna(ref_ant, delay_model=dm).location
                                    for dm in ant_models.values()] + [ref_ant.location])

    @property
    def tropospheric_model(self):
        """Unique identifier of tropospheric model, or 'None' for no correction."""
        return (self._tropospheric_delay.model_id  # pylint: disable=no-member
                if self._tropospheric_delay else 'None')

    @property
    def ant_locations(self):
        """Locations of *A* antennas, in the same order as `inputs`."""
        return self._locations[:-1]

    @property
    def ref_location(self):
        """Location of reference antenna."""
        return self._locations[-1]

    @property
    @u.quantity_input
    def max_delay(self) -> u.s:
        """The maximum (absolute) delay achievable in the array."""
        # Worst case is wavefront moving along baseline connecting ant to ref
        max_delay_per_ant = np.linalg.norm(self._params[:, :3], axis=1)
        # Pick largest fixed delay
        max_delay_per_ant += self._params[:, 3:5].max(axis=1)
        # Worst case for NIAO is looking at the horizon
        max_delay_per_ant += self._params[:, 5]
        return max(max_delay_per_ant) if self.ant_models else 0.0 * u.s

    @property
    def description(self):
        """Complete string representation of object that allows reconstruction."""
        descr = {'ref_ant': self.ref_ant.description,
                 'sky_centre_freq': self.sky_centre_freq.to_value(u.Hz),
                 'extra_correction': self.extra_correction.to_value(u.s),
                 'tropospheric_model': self.tropospheric_model,
                 'ant_models': {ant: model.description
                                for ant, model in self.ant_models.items()}}
        return json.dumps(descr, sort_keys=True)

    _DELAY_PARAMETERS_DOCSTRING = """
        Parameters
        ----------
        target : :class:`Target`
            Target providing direction for geometric delays
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equivalent
            Timestamp(s) when wavefront from target passes reference position, shape T
        offset : dict, optional
            Keyword arguments for :meth:`Target.plane_to_sphere` to offset
            delay centre relative to target (see method for details)
        pressure : :class:`~astropy.units.Quantity`, optional
            Total barometric pressure at surface, broadcastable to shape (A, prod(T))
        temperature : :class:`~astropy.units.Quantity`, optional
            Ambient air temperature at surface, broadcastable to shape (A, prod(T)).
            If the relative humidity is positive, the temperature has to be set.
        relative_humidity : :class:`~astropy.units.Quantity` or array-like, optional
            Relative humidity at surface, as a fraction in range [0, 1],
            broadcastable to shape (A, prod(T))
        """.strip()

    @u.quantity_input(equivalencies=u.temperature())
    def delays(self, target, timestamp, offset=None,
               pressure: u.hPa = 0 * u.hPa, temperature: u.deg_C = NO_TEMPERATURE,
               relative_humidity: u.dimensionless_unscaled = 0) -> u.s:
        """Calculate delays for all timestamps and inputs for a given target.

        These delays include all geometric effects (also non-intersecting axis
        offsets) and known fixed/cable delays, but not the :attr:`extra_correction`
        needed to make delay corrections strictly positive.

        {Parameters}

        Returns
        -------
        delays : :class:`~astropy.units.Quantity`, shape (2 * A,) + T
            Delays for *2A* correlator inputs and timestamps with shape T, with
            ordering on the first axis matching the labels in :attr:`inputs`

        Raises
        ------
        ValueError
            If the relative humidity is positive but temperature is unspecified
        """
        # Ensure a single consistent timestamp in the case of "now"
        time = Timestamp(timestamp).time
        T = time.shape
        # Manually broadcast both time and location to shape (A + 1, prod(T))
        # XXX Astropy 4.2 has proper broadcasting support (at least for obstime)
        time = time.ravel()
        time_idx, location_idx = np.meshgrid(range(len(time)), range(len(self._locations)))
        time = time.take(time_idx)
        locations = self._locations.take(location_idx)
        # Obtain (az, el) pointings per location and timestamp => shape (A + 1, prod(T))
        if not offset:
            azel = target.azel(time, locations)
            az = azel.az.rad
            el = azel.alt.rad
        else:
            coord_system = offset.get('coord_system', 'azel')
            if coord_system == 'radec':
                ra, dec = target.plane_to_sphere(timestamp=time, antenna=locations, **offset)
                # XXX This target is vectorised (contrary to popular belief) by having an
                # array-valued SkyCoord inside its FixedBody, so .azel() does the right thing.
                # It is probably better to support this explicitly somehow.
                offset_target = Target.from_radec(ra, dec)
                azel = offset_target.azel(time, locations)
                az = azel.az.rad
                el = azel.alt.rad
            else:
                az, el = target.plane_to_sphere(timestamp=time, antenna=locations, **offset)
        # Elevations of antennas proper, shape (A, prod(T))
        elevations = el[:-1] * u.rad
        # Obtain target direction as seen from reference location => shape (3, prod(T))
        target_dir = np.array(azel_to_enu(az[-1], el[-1]))
        # Split up delay model parameters into constituent parts (unit = seconds)
        enu_offset = self._params[:, :3]  # shape (A, 3)
        fixed_path_length = self._params[:, 3:5]  # shape (A, 2)
        niao = self._params[:, 5:6]  # shape (A, 1)
        # Combine all delays per antenna (geometric, NIAO, troposphere) => shape (A, prod(T))
        ant_delays = enu_offset @ -target_dir
        ant_delays -= niao * np.cos(elevations)
        if self._tropospheric_delay:
            if temperature == NO_TEMPERATURE and relative_humidity > 0:
                raise ValueError(f'The relative humidity is set to {relative_humidity} '
                                 'but the temperature has not been specified')
            ant_delays += self._tropospheric_delay(pressure, temperature, relative_humidity,
                                                   elevations, time[:-1])
        # Expand delays per antenna to delays per input => shape (A, 2, prod(T))
        input_delays = np.stack([ant_delays, ant_delays], axis=1)
        input_delays += fixed_path_length[..., np.newaxis]
        # Collapse input dimensions and restore time dimensions => shape (2 * A,) + T
        return input_delays.reshape((-1,) + T)

    @u.quantity_input(equivalencies=u.temperature())
    def corrections(self, target, timestamp=None, offset=None,
                    pressure: u.hPa = 0 * u.hPa, temperature: u.deg_C = NO_TEMPERATURE,
                    relative_humidity: u.dimensionless_unscaled = 0):
        """Delay and phase corrections for a given target and timestamp(s).

        Calculate delay and phase corrections for the direction towards
        `target` at `timestamp`. Both delay (aka phase slope across frequency)
        and phase (aka phase offset or fringe phase) corrections are provided,
        and their derivatives with respect to time (delay rate and fringe rate,
        respectively). The derivatives allow linear interpolation of delay
        and phase if a sequence of timestamps is provided.

        {Parameters}

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

        Raises
        ------
        ValueError
            If the relative humidity is positive but temperature is unspecified
        """
        time = Timestamp(timestamp).time
        T = time.shape
        # Ensure that times are at least 1-D (and delays 2-D) so that we can calculate deltas
        # XXX Astropy 4.2 supports np.atleast_1d(time)
        if time.isscalar:
            time = time[np.newaxis]
        delays = self.delays(target, time, offset, pressure, temperature, relative_humidity)
        delay_corrections = self.extra_correction - delays
        # The phase term is (-2 pi freq delay) so the correction is (+2 pi freq delay)
        turns = (self.sky_centre_freq * delays).decompose()
        phase_corrections = Angle(2. * np.pi * u.rad) * turns
        delta_time = (time[1:] - time[:-1]).to(u.s)
        delta_delay_corrections = delay_corrections[:, 1:] - delay_corrections[:, :-1]
        delay_rate_corrections = delta_delay_corrections / delta_time
        delta_phase_corrections = phase_corrections[:, 1:] - phase_corrections[:, :-1]
        fringe_rate_corrections = delta_phase_corrections / delta_time
        return (
            # Restore time dimensions to recover scalar times
            dict(zip(self.inputs, delay_corrections.reshape((-1,) + T))),
            dict(zip(self.inputs, phase_corrections.reshape((-1,) + T))),
            dict(zip(self.inputs, delay_rate_corrections)),
            dict(zip(self.inputs, fringe_rate_corrections)),
        )

    delays.__doc__ = delays.__doc__.format(Parameters=_DELAY_PARAMETERS_DOCSTRING)
    corrections.__doc__ = corrections.__doc__.format(Parameters=_DELAY_PARAMETERS_DOCSTRING)
