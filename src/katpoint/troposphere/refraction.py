################################################################################
# Copyright (c) 2009-2022, National Research Foundation (SARAO)
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

"""Tropospheric refraction model.

This predicts the refractive bending due to neutral gas in the troposphere
and stratosphere as a function of elevation angle, based on surface weather
measurements.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class HaystackRefraction:

    @classmethod
    def refract(cls, el, temperature_C, pressure_hPa, humidity_percent):
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
        refracted_el : float or array
            Refracted elevation angle(s), in radians

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
        a = 40.0
        b = 2.7
        c = 4.0
        d = 42.5
        e = 0.4
        f = 2.64
        g = 0.57295787e-4

        # Compute SN (surface refractivity)
        # (via dewpoint and water vapor partial pressure? [LS])
        rhumi = (100.0 - humidity_percent) * 0.9
        dewpt = temperature_C - rhumi * (
            0.136667 + rhumi * 1.33333e-3 + temperature_C * 1.5e-3
        )
        pp = p[0] + p[1] * dewpt + p[2] * dewpt**2 + p[3] * dewpt**3 + p[4] * dewpt**4
        temperature_K = temperature_C + 273.0
        # This looks like Smith & Weintraub (1953) or Crane (1976) [LS]
        sn = 77.6 * (pressure_hPa + (4810.0 * cvt * pp) / temperature_K) / temperature_K

        # Compute refraction at elevation
        # (clipped at 1 degree to avoid cot(el) blow-up at horizon)
        el_deg = np.clip(np.degrees(el), 1.0, 90.0)
        aphi = a / ((el_deg + b) ** c)
        dele = -d / ((el_deg + e) ** f)
        zenith_angle = np.radians(90.0 - el_deg)
        bphi = g * (np.tan(zenith_angle) + dele)
        # Threw out an (el < 0.01) check here,
        # which will never succeed because el is clipped to be above 1.0 [LS]

        return el + np.radians(bphi * sn - aphi)

    @classmethod
    def unrefract(cls, refracted_el, temperature_C, pressure_hPa, humidity_percent):
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
        # Maximum difference between input elevation and
        # refraction-corrected version of final output elevation
        tolerance = np.radians(0.01 / 3600)
        # Assume offset from corrected el is similar to offset from uncorrected el
        # -> get lower bound on desired el
        close_offset = cls.refract(
            refracted_el, temperature_C, pressure_hPa, humidity_percent
        ) - refracted_el
        lower = refracted_el - 4 * np.abs(close_offset)
        # We know that corrected el > uncorrected el (mostly)
        # -> this becomes upper bound on desired el
        upper = refracted_el + np.radians(1.0 / 3600.0)
        # Do binary search for desired el within this range (but cap iterations
        # in case of a mishap). This assumes that refraction-corrected elevation
        # is monotone function of uncorrected elevation.
        for iteration in range(40):
            el = 0.5 * (lower + upper)
            test_el = cls.refract(el, temperature_C, pressure_hPa, humidity_percent)
            if np.all(np.abs(test_el - refracted_el) < tolerance):
                break
            lower = np.where(test_el < refracted_el, el, lower)
            upper = np.where(test_el > refracted_el, el, upper)
        else:
            logger.warning(
                "Reverse refraction correction did not converge in "
                "%d iterations - elevation differs by at most %f arcsecs",
                iteration + 1,
                np.degrees(np.abs(test_el - refracted_el).max()) * 3600.0,
            )
        return el if el.ndim else el.item()


_REFRACTION = {
    "HaystackRefraction": HaystackRefraction,
}


class TroposphericRefraction:
    """Correct pointing for refractive bending in atmosphere.

    This uses the specified refraction model to calculate a correction to a
    given elevation angle to account for refractive bending in the atmosphere,
    based on surface weather measurements. The refraction correction can also
    be undone, usually to refer the actual antenna position to the coordinate
    frame before corrections were applied.

    Parameters
    ----------
    model_id : str, optional
        Unique identifier of tropospheric model (defaults to best model)

    Raises
    ------
    ValueError
        If the specified refraction model is unknown
    """

    def __init__(self, model_id="HaystackRefraction"):
        try:
            model = _REFRACTION[model_id]
        except KeyError as err:
            raise ValueError(
                f"Unknown tropospheric refraction model {model_id!r}, "
                f"available ones are {list(_REFRACTION.keys())}"
            ) from err
        # These will effectively be read-only attributes because setattr is disabled
        super().__setattr__("model_id", model_id)
        super().__setattr__("_model", model)

    def __setattr__(self, name, value):
        """Prevent modification of attributes (the model is read-only)."""
        raise AttributeError("Tropospheric refraction models are immutable")

    def __repr__(self):
        """Short human-friendly string representation of object."""
        return f"<katpoint.TroposphericRefraction {self.model_id!r} at {id(self):#x}>"

    def __eq__(self, other):
        """Equality comparison operator."""
        return (
            isinstance(other, TroposphericRefraction)
            and (self.model_id == other.model_id)
        )

    def __hash__(self):
        """Compute hash on underlying model name, just like equality operator."""
        return hash((self.__class__, self.model_id))

    def refract(self, el, temperature_C, pressure_hPa, humidity_percent):
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
        return self._model.refract(el, temperature_C, pressure_hPa, humidity_percent)

    def unrefract(self, el, temperature_C, pressure_hPa, humidity_percent):
        """Remove refraction correction from elevation angle.

        This undoes a refraction correction that resulted in the given elevation
        angle. It is the inverse of :meth:`refract`.

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
        return self._model.unrefract(el, temperature_C, pressure_hPa, humidity_percent)
