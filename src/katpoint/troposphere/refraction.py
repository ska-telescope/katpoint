################################################################################
# Copyright (c) 2009-2024, National Research Foundation (SARAO)
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
from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle

logger = logging.getLogger(__name__)


class _RefractionModel:
    """A basic tropospheric refraction model operating on elevation angles."""


class HaystackRefraction(_RefractionModel):
    """Refraction model of the MIT Haystack Pointing System."""

    @classmethod
    @u.quantity_input(equivalencies=u.temperature())
    def refract(
        cls,
        elevation: u.deg,
        pressure: u.hPa,
        temperature: u.deg_C,
        relative_humidity: u.dimensionless_unscaled,
    ) -> u.deg:
        """Calculate refracted elevation angle from unrefracted `elevation`.

        Parameters
        ----------
        elevation : :class:`~astropy.coordinates.Angle`-like
            Unrefracted / topocentric / vacuum elevation angle
        pressure : :class:`~astropy.units.Quantity`
            Total barometric pressure at surface
        temperature : :class:`~astropy.units.Quantity`
            Ambient air temperature at surface
        relative_humidity : :class:`~astropy.units.Quantity` or float or array
            Relative humidity at surface, as a fraction in range [0, 1]

        Returns
        -------
        refracted_elevation : :class:`~astropy.coordinates.Angle`
            Refracted / observed / surface elevation angle

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
        `<https://doi.org/10.21236/ad0641603>`_ or
        `<https://apps.dtic.mil/sti/tr/pdf/AD0641603.pdf>`_
        .. [IH1963] W.R. Iliff, J.M. Holt, "Use of Surface Refractivity in the
        Empirical Prediction of Total Atmospheric Refraction," Journal of Research
        of the National Bureau of Standards--D. Radio Propagation, vol. 67D,
        no. 1, Jan 1963, `<https://doi.org/10.6028/jres.067d.006>`_
        """
        elevation = Angle(elevation)
        pressure_hPa = pressure.value
        temperature_C = temperature.value

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
        rhumi = 100.0 * (1.0 - relative_humidity) * 0.9
        dewpt = temperature_C - rhumi * (
            0.136667 + rhumi * 1.33333e-3 + temperature_C * 1.5e-3
        )
        pp = (
            p[0]
            + p[1] * dewpt
            + p[2] * dewpt**2
            + p[3] * dewpt**3
            + p[4] * dewpt**4
        )
        # Left this inaccurate conversion in case coefficients were fitted with it [LS]
        temperature_K = temperature_C + 273.0
        # This looks like Smith & Weintraub (1953) or Crane (1976) [LS]
        sn = 77.6 * (pressure_hPa + (4810.0 * cvt * pp) / temperature_K) / temperature_K

        # Compute refraction at elevation
        # (clipped at 1 degree to avoid cot(el) blow-up at horizon)
        el = np.clip(elevation, 1.0 * u.deg, 90.0 * u.deg)
        aphi = a / ((el.deg + b) ** c)
        dele = -d / ((el.deg + e) ** f)
        zenith_angle = 90.0 * u.deg - el
        bphi = g * (np.tan(zenith_angle) + dele)
        # Threw out an (el < 0.01) check here,
        # which will never succeed because el is clipped to be above 1.0 [LS]

        return elevation + (bphi * sn - aphi) * u.deg

    @classmethod
    @u.quantity_input(equivalencies=u.temperature())
    def unrefract(
        cls,
        refracted_elevation: u.deg,
        pressure: u.hPa,
        temperature: u.deg_C,
        relative_humidity: u.dimensionless_unscaled,
    ) -> u.deg:
        """Calculate unrefracted elevation angle from `refracted_el`.

        This undoes a refraction correction that resulted in the given elevation
        angle. It is the inverse of :meth:`refract`.

        Parameters
        ----------
        refracted_elevation : :class:`~astropy.coordinates.Angle`-like
            Refracted / observed / surface elevation angle
        pressure : :class:`~astropy.units.Quantity`
            Total barometric pressure at surface
        temperature : :class:`~astropy.units.Quantity`
            Ambient air temperature at surface
        relative_humidity : :class:`~astropy.units.Quantity` or float or array
            Relative humidity at surface, as a fraction in range [0, 1]

        Returns
        -------
        elevation : :class:`~astropy.coordinates.Angle`
            Unrefracted / topocentric / vacuum elevation angle
        """
        refracted_elevation = Angle(refracted_elevation)
        # Maximum difference between input elevation and
        # refraction-corrected version of final output elevation
        tolerance = 10 * u.mas
        # Assume offset from corrected el is similar to offset from uncorrected el
        # -> get lower bound on desired el
        close_offset = (
            cls.refract(refracted_elevation, pressure, temperature, relative_humidity)
            - refracted_elevation
        )
        lower = refracted_elevation - 4 * np.abs(close_offset)
        # We know that corrected el > uncorrected el (mostly)
        # -> this becomes upper bound on desired el
        upper = refracted_elevation + 1 * u.arcsec
        # Do binary search for desired el within this range (but cap iterations
        # in case of a mishap). This assumes that refraction-corrected elevation
        # is monotone function of uncorrected elevation.
        for iteration in range(40):
            el = 0.5 * (lower + upper)
            test_el = cls.refract(el, pressure, temperature, relative_humidity)
            if np.all(np.abs(test_el - refracted_elevation) < tolerance):
                break
            lower = np.where(test_el < refracted_elevation, el, lower)
            upper = np.where(test_el > refracted_elevation, el, upper)
        else:
            logger.warning(
                "Reverse refraction correction did not converge in "
                "%d iterations - elevation differs by at most %f arcsecs",
                iteration + 1,
                np.abs(test_el - refracted_elevation).max().to_value(u.arcsec),
            )
        return el if el.ndim else el.item()


@dataclass(frozen=True)
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

    model_id: str = "HaystackRefraction"
    _model: _RefractionModel = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Initialise underlying refraction `_model` from `model_id`."""
        models = {cls.__name__: cls for cls in _RefractionModel.__subclasses__()}
        try:
            model = models[self.model_id]
        except KeyError as err:
            raise ValueError(
                f"Unknown tropospheric refraction model {self.model_id!r}, "
                f"available ones are {list(models.keys())}"
            ) from err
        # Set attribute on base class because this class is frozen
        super().__setattr__("_model", model)

    @u.quantity_input(equivalencies=u.temperature())
    def refract(
        self,
        elevation: u.deg,
        pressure: u.hPa,
        temperature: u.deg_C,
        relative_humidity: u.dimensionless_unscaled,
    ) -> u.deg:
        """Apply refraction correction to elevation angle.

        Parameters
        ----------
        elevation : :class:`~astropy.coordinates.Angle`
            Unrefracted / topocentric / vacuum elevation angle
        pressure : :class:`~astropy.units.Quantity`
            Total barometric pressure at surface
        temperature : :class:`~astropy.units.Quantity`
            Ambient air temperature at surface
        relative_humidity : :class:`~astropy.units.Quantity` or float or array
            Relative humidity at surface, as a fraction in range [0, 1]

        Returns
        -------
        refracted_elevation : :class:`~astropy.coordinates.Angle`
            Refracted / observed / surface elevation angle
        """
        return self._model.refract(elevation, pressure, temperature, relative_humidity)

    @u.quantity_input(equivalencies=u.temperature())
    def unrefract(
        self,
        refracted_elevation: u.deg,
        pressure: u.hPa,
        temperature: u.deg_C,
        relative_humidity: u.dimensionless_unscaled,
    ) -> u.deg:
        """Remove refraction correction from elevation angle.

        This undoes a refraction correction that resulted in the given elevation
        angle. It is the inverse of :meth:`refract`.

        Parameters
        ----------
        refracted_elevation : :class:`~astropy.coordinates.Angle`
            Refracted / observed / surface elevation angle
        pressure : :class:`~astropy.units.Quantity`
            Total barometric pressure at surface
        temperature : :class:`~astropy.units.Quantity`
            Ambient air temperature at surface
        relative_humidity : :class:`~astropy.units.Quantity` or float or array
            Relative humidity at surface, as a fraction in range [0, 1]

        Returns
        -------
        elevation : :class:`~astropy.coordinates.Angle`
            Unrefracted / topocentric / vacuum elevation angle
        """
        return self._model.unrefract(
            refracted_elevation, pressure, temperature, relative_humidity
        )
