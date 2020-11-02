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

"""Refraction correction.

This implements correction for refractive bending in the atmosphere.
"""

import logging

import numpy as np
import astropy.units as u
import astropy.constants as const


logger = logging.getLogger(__name__)


def refraction_offset_vlbi(el, temperature_C, pressure_hPa, humidity_percent):
    """Calculate refraction correction using model in VLBI Field System.

    This uses the refraction model in the VLBI Field System to calculate a
    correction to a given elevation angle to account for refractive bending in
    the atmosphere, based on surface weather measurements. Each input parameter
    can either be a scalar value or an array of values, as long as all arrays
    are of the same shape.

    Parameters
    ----------
    el : float or array
        Requested elevation angle(s), in radians
    temperature_C : float or array
        Ambient air temperature at surface, in degrees Celsius
    pressure_hPa : float or array
        Total barometric pressure at surface, in hectopascal (hPa) or millibars
    humidity_percent : float or array
        Relative humidity at surface, as a percentage in range [0, 100]

    Returns
    -------
    el_offset : float or array
        Refraction offset(s) in radians, which needs to be *added* to
        elevation angle(s) to correct it

    Notes
    -----
    The code is based on poclb/refrwn.c in Field System version 9.9.2, which
    was added on 2006-11-15. This is a C version (with typos fixed) of the
    Fortran version in polb/refr.f. As noted in the Field System
    documentation [Him1993b]_, the refraction model originated with the Haystack
    pointing system. A description of the model can be found in [Clark1966]_,
    which in turn references [IH1963]_ as the ultimate source.

    References
    ----------
    .. [Him1993b] E. Himwich, "Station Programs," Mark IV Field System Reference
       Manual, Version 8.2, 1 September 1993.
    .. [Clark1966] C.A. Clark, "Haystack Pointing System: Radar Coordinate
       Correction," Technical Note 1966-56, Lincoln Laboratory, MIT, 1966,
       `<https://doi.org/10.21236/ad0641603>`_
    .. [IH1963] W.R. Iliff, J.M. Holt, "Use of Surface Refractivity in the
       Empirical Prediction of Total Atmospheric Refraction," Journal of Research
       of the National Bureau of Standards--D. Radio Propagation, vol. 67D,
       no. 1, Jan 1963, `<https://doi.org/10.6028/jres.067d.006>`_
    """
    p = (0.458675e1, 0.322009e0, 0.103452e-1, 0.274777e-3, 0.157115e-5)
    cvt = 1.33289
    a = 40.
    b = 2.7
    c = 4.
    d = 42.5
    e = 0.4
    f = 2.64
    g = 0.57295787e-4

    # Compute SN (surface refractivity) (via dewpoint and water vapor partial pressure? [LS])
    rhumi = (100. - humidity_percent) * 0.9
    dewpt = temperature_C - rhumi * (0.136667 + rhumi * 1.33333e-3 + temperature_C * 1.5e-3)
    pp = p[0] + p[1] * dewpt + p[2] * dewpt ** 2 + p[3] * dewpt ** 3 + p[4] * dewpt ** 4
    temperature_K = temperature_C + 273.
    # This looks like Smith & Weintraub (1953) or Crane (1976) [LS]
    sn = 77.6 * (pressure_hPa + (4810.0 * cvt * pp) / temperature_K) / temperature_K

    # Compute refraction at elevation (clipped at 1 degree to avoid cot(el) blow-up at horizon)
    el_deg = np.clip(np.degrees(el), 1.0, 90.0)
    aphi = a / ((el_deg + b) ** c)
    dele = -d / ((el_deg + e) ** f)
    zenith_angle = np.radians(90. - el_deg)
    bphi = g * (np.tan(zenith_angle) + dele)
    # Threw out an (el < 0.01) check here, which will never succeed because el is clipped to be above 1.0 [LS]

    return np.radians(bphi * sn - aphi)


class RefractionCorrection:
    """Correct pointing for refractive bending in atmosphere.

    This uses the specified refraction model to calculate a correction to a
    given elevation angle to account for refractive bending in the atmosphere,
    based on surface weather measurements. The refraction correction can also
    be undone, usually to refer the actual antenna position to the coordinate
    frame before corrections were applied.

    Parameters
    ----------
    model : string, optional
        Name of refraction model to use

    Raises
    ------
    ValueError
        If the specified refraction model is unknown
    """

    def __init__(self, model='VLBI Field System'):
        self.models = {'VLBI Field System': refraction_offset_vlbi}
        try:
            self.offset = self.models[model]
        except KeyError:
            raise ValueError("Unknown refraction correction model '%s' - should be one of %s" %
                             (model, self.models.keys()))
        self.model = model

    def __repr__(self):
        """Short human-friendly string representation of refraction correction object."""
        return "<katpoint.RefractionCorrection model='%s' at 0x%x>" % (self.model, id(self))

    def __eq__(self, other):
        """Equality comparison operator."""
        return isinstance(other, RefractionCorrection) and (self.model == other.model)

    def __ne__(self, other):
        """Inequality comparison operator."""
        return not (self == other)

    def __hash__(self):
        """Base hash on underlying model name, just like equality operator."""
        return hash((self.__class__, self.model))

    def apply(self, el, temperature_C, pressure_hPa, humidity_percent):
        """Apply refraction correction to elevation angle.

        Each input parameter can either be a scalar value or an array of values,
        as long as all arrays are of the same shape.

        Parameters
        ----------
        el : float or array
            Requested elevation angle(s), in radians
        temperature_C : float or array
            Ambient air temperature at surface, in degrees Celsius
        pressure_hPa : float or array
            Total barometric pressure at surface, in hectopascal (hPa) or millibars
        humidity_percent : float or array
            Relative humidity at surface, as a percentage in range [0, 100]

        Returns
        -------
        refracted_el : float or array
            Elevation angle(s), corrected for refraction, in radians
        """
        return el + self.offset(el, temperature_C, pressure_hPa, humidity_percent)

    def reverse(self, refracted_el, temperature_C, pressure_hPa, humidity_percent):
        """Remove refraction correction from elevation angle.

        This undoes a refraction correction that resulted in the given elevation
        angle. It is the inverse of :meth:`apply`.

        Parameters
        ----------
        refracted_el : float or array
            Elevation angle(s), corrected for refraction, in radians
        temperature_C : float or array
            Ambient air temperature at surface, in degrees Celsius
        pressure_hPa : float or array
            Total barometric pressure at surface, in hectopascal (hPa) or millibars
        humidity_percent : float or array
            Relative humidity at surface, as a percentage in range [0, 100]

        Returns
        -------
        el : float or array
            Elevation angle(s) before refraction correction, in radians
        """
        # Maximum difference between input elevation and refraction-corrected version of final output elevation
        tolerance = np.radians(0.01 / 3600)
        # Assume offset from corrected el is similar to offset from uncorrected el -> get lower bound on desired el
        close_offset = self.offset(refracted_el, temperature_C, pressure_hPa, humidity_percent)
        lower = refracted_el - 4 * np.abs(close_offset)
        # We know that corrected el > uncorrected el (mostly) -> this becomes upper bound on desired el
        upper = refracted_el + np.radians(1. / 3600.)
        # Do binary search for desired el within this range (but cap iterations in case of a mishap)
        # This assumes that refraction-corrected elevation is monotone function of uncorrected elevation
        for iteration in range(40):
            el = 0.5 * (lower + upper)
            test_el = self.apply(el, temperature_C, pressure_hPa, humidity_percent)
            if np.all(np.abs(test_el - refracted_el) < tolerance):
                break
            lower = np.where(test_el < refracted_el, el, lower)
            upper = np.where(test_el > refracted_el, el, upper)
        else:
            logger.warning('Reverse refraction correction did not converge in '
                           '%d iterations - elevation differs by at most %f arcsecs',
                           iteration + 1, np.degrees(np.abs(test_el - refracted_el).max()) * 3600.)
        return el if el.ndim else el.item()


class SaastamoinenZenithDelay:
    """Zenith delay due to the neutral gas in the troposphere and stratosphere.

    This provides separate methods for the "dry" (hydrostatic) and "wet"
    (non-hydrostatic) components of the atmosphere.

    Parameters
    ----------
    location : `~astropy.coordinates.EarthLocation`
        Location on Earth of observer (used to correct local gravity)

    Notes
    -----
    This is a direct translation of the SASTD and SASTW subroutines in the
    atmospheric module (catmm.f) of Calc 11. It is based on the formulas of
    Saastamoinen [1]_ as implemented by Davis et al [2]_. The saturation
    water vapour pressure is calculated by the venerable but still practical
    August-Roche-Magnus formula as discussed in [3]_.

    References
    ----------
    .. [1] J. Saastamoinen, “Atmospheric correction for the troposphere and
       stratosphere in radio ranging satellites,” in The Use of Artificial
       Satellites for Geodesy (Geophysical Monograph Series), edited by
       S. W. Henriksen et al, Washington, D.C., vol. 15, pp. 247-251, 1972.
       DOI: 10.1029/GM015p0247

    .. [2] J. L. Davis, T. A. Herring, I. I. Shapiro, A. E. E. Rogers, and
       G. Elgered, “Geodesy by radio interferometry: Effects of atmospheric
       modeling errors on estimates of baseline length,” Radio Science,
       vol. 20, no. 6, pp. 1593-1607, 1985. DOI: 10.1029/rs020i006p01593

    .. [3] F. W. Murray, “On the computation of saturation vapor pressure,”
        Journal of Applied Meteorology, vol. 6, no. 1, pp. 203-204, Feb 1967.
        DOI: 10.1175/1520-0450(1967)006<0203:OTCOSV>2.0.CO;2
    """

    def __init__(self, location):
        # Reduce local gravity to the value at the centroid of the atmospheric column,
        # which depends on the location of the observer
        latitude_rad = location.lat.rad
        height_km = location.height.to_value(u.km)
        self._gravity_correction = 1. - 0.00266 * np.cos(2. * latitude_rad) - 0.00028 * height_km

    def hydrostatic(self, temperature, pressure, humidity):
        """Zenith delay due to "dry" (hydrostatic) component of the atmosphere.

        Parameters
        ----------
        temperature : :class:`~astropy.units.Quantity`, float or array
            Ambient air temperature at surface (ignored)
        pressure : :class:`~astropy.units.Quantity`, float or array
            Total barometric pressure at surface (hectopascal if not a `Quantity`)
        humidity : float or array
            Relative humidity at surface (ignored)

        Returns
        -------
        delay : :class:`~astropy.units.Quantity`
            Zenith delay due to hydrostatic component, in seconds
        """
        excess_path_per_hPa = 0.0022768 * u.m
        pressure_hPa = (pressure << u.hectopascal).value
        return excess_path_per_hPa * pressure_hPa / self._gravity_correction / const.c

    def wet(self, temperature, pressure, humidity):
        """Zenith delay due to "wet" (non-hydrostatic) component of atmosphere.

        Parameters
        ----------
        temperature : :class:`~astropy.units.Quantity`, float or array
            Ambient air temperature at surface (degrees Celsius if not a `Quantity`)
        pressure : :class:`~astropy.units.Quantity`, float or array
            Total barometric pressure at surface (ignored)
        humidity : float or array
            Relative humidity at surface, as a fraction in range [0, 1]

        Returns
        -------
        delay : :class:`~astropy.units.Quantity`
            Zenith delay due to non-hydrostatic component, in seconds
        """
        temperature_C = (temperature << u.deg_C).value
        # This resembles the version of the August-Roche-Magnus formula in Murray (1967)
        saturation_pressure_hPa = 6.11 * np.exp(17.269 * temperature_C / (temperature_C + 237.3))
        # The Tetens (1930) version, which serves as the reference, is
        # saturation_pressure_hPa = 10 ** (7.5 * temperature_C / (temperature_C + 237.3) + 0.7858)
        partial_pressure_hPa = humidity * saturation_pressure_hPa
        # Saastamoinen suggested 273.2, Astropy units has 273.15 but Murray and CODATA likes 273.16
        temperature_K = temperature_C + 273.16
        excess_path_per_hPa = 0.002277 * (1255. / temperature_K + 0.05) * u.m
        return excess_path_per_hPa * partial_pressure_hPa / const.c


ZENITH_DELAY = {'SaastamoinenZD': SaastamoinenZenithDelay}


class GlobalMappingFunction:
    """"""

    def __init__(self, location):
        self.location = location

    def hydrostatic(self, elevation, timestamp):
        return 1.0

    def wet(self, elevation, timestamp):
        return 1.0


MAPPING_FUNCTION = {'GlobalMF': GlobalMappingFunction}


class TroposphericDelay:
    """"""

    def __init__(self, location, model_id='SaastamoinenZD-GlobalMF'):
        # These will effectively be read-only attributes because setattr is disabled
        super().__setattr__('location', location)
        super().__setattr__('model_id', model_id)
        # Parse model identifier string
        model_parts = model_id.split('-')
        if len(model_parts) == 2:
            model_parts.append('total')
        if len(model_parts) != 3:
            raise ValueError(f"Format for tropospheric delay model ID is '<zenith delay>-"
                             f"<mapping function>[-<hydrostatic/wet>]', not {model_id:!r}")

        def get(mapping, key, name):
            try:
                return mapping[key]
            except KeyError as err:
                raise ValueError(f"Tropospheric delay model {model_id:!r} has unknown {name} "
                                 f"{key:!r}, available ones are {list(mapping.keys())}") from err

        zenith_delay = get(ZENITH_DELAY, model_parts[0], 'zenith delay function')(location)
        mapping_function = get(MAPPING_FUNCTION, model_parts[1], 'mapping function')(location)

        def hydrostatic(t, p, h, el, ts):
            return zenith_delay.hydrostatic(t, p, h) * mapping_function.hydrostatic(el, ts)

        def wet(t, p, h, el, ts):
            return zenith_delay.wet(t, p, h) * mapping_function.wet(el, ts)

        def total(t, p, h, el, ts):
            return hydrostatic(t, p, h, el, ts) + wet(t, p, h, el, ts)

        model_types = {'hydrostatic': hydrostatic, 'wet': wet, 'total': total}
        super().__setattr__('_delay', get(model_types, model_parts[2], 'type'))

    def __setattr__(self, name, value):
        """Prevent modification of attributes (the model is read-only)."""
        raise AttributeError('Tropospheric delay models are immutable')

    def __call__(self, temperature, pressure, humidity, elevation, timestamp):
        """"""
        return self._delay(temperature, pressure, humidity, elevation, timestamp)
