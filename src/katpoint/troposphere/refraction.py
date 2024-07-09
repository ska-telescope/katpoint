################################################################################
# Copyright (c) 2009-2013,2016-2019,2023-2024, National Research Foundation (SARAO)
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
from typing import Type, TypeVar

import astropy.units as u
import erfa
import numpy as np
from astropy.coordinates import AltAz, UnitSphericalRepresentation

logger = logging.getLogger(__name__)


class _RefractionModel:
    """A basic tropospheric refraction model operating on elevation angles.

    This is considered a low-level class that operates on floats instead of
    Astropy Quantities, sort of like PyERFA.
    """

    @classmethod
    def refract(
        cls,
        elevation_rad,
        pressure_hPa,
        temperature_C,
        humidity_frac,
    ):
        """Calculate refracted elevation angle from unrefracted elevation.

        Parameters
        ----------
        elevation_rad : float or array
            Unrefracted / topocentric / vacuum elevation angle, in radians
        pressure_hPa : float or array
            Total barometric pressure at surface, in hectopascal or millibars
        temperature_C : float or array
            Ambient air temperature at surface, in degrees Celsius
        humidity_frac : float or array
            Relative humidity at surface, as a fraction in range [0, 1]

        Returns
        -------
        refracted_el_rad : float or array
            Refracted / observed / surface elevation angle, in radians
        """
        raise NotImplementedError

    @classmethod
    def unrefract(
        cls,
        refracted_el_rad,
        pressure_hPa,
        temperature_C,
        humidity_frac,
    ):
        """Calculate unrefracted elevation angle from refracted elevation.

        This undoes a refraction correction that resulted in the given elevation
        angle. It is the inverse of :meth:`refract`.

        Parameters
        ----------
        refracted_el_rad : float or array
            Refracted / observed / surface elevation angle, in radians
        pressure_hPa : float or array
            Total barometric pressure at surface, in hectopascal or millibars
        temperature_C : float or array
            Ambient air temperature at surface, in degrees Celsius
        humidity_frac : float or array
            Relative humidity at surface, as a fraction in range [0, 1]

        Returns
        -------
        elevation_rad : float or array
            Unrefracted / topocentric / vacuum elevation angle, in radians
        """
        raise NotImplementedError


_SomeRefractionModel = TypeVar("_SomeRefractionModel", bound=_RefractionModel)


class HaystackRefraction(_RefractionModel):
    """Refraction model of the 1960's MIT Haystack Pointing System.

    The `refract` method is the main routine in this model, while the
    `unrefract` method implements a generic function inverse.

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

    @classmethod
    def refract(  # noqa: D102 (docstring inherited from base class)
        cls,
        elevation_rad,
        pressure_hPa,
        temperature_C,
        humidity_frac,
    ):
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
        rhumi = 100.0 * (1.0 - humidity_frac) * 0.9
        dewpt = temperature_C - rhumi * (
            0.136667 + rhumi * 1.33333e-3 + temperature_C * 1.5e-3
        )
        pp = p[0] + p[1] * dewpt + p[2] * dewpt**2 + p[3] * dewpt**3 + p[4] * dewpt**4
        # Left this inaccurate conversion in case coefficients were fitted with it [LS]
        temperature_K = temperature_C + 273.0
        # This looks like Smith & Weintraub (1953) or Crane (1976) [LS]
        sn = 77.6 * (pressure_hPa + (4810.0 * cvt * pp) / temperature_K) / temperature_K

        # Compute refraction at elevation
        # (clipped at 1 degree to avoid cot(el) blow-up at horizon)
        el_deg = np.clip(np.degrees(elevation_rad), 1.0, 90.0)
        aphi = a / ((el_deg + b) ** c)
        dele = -d / ((el_deg + e) ** f)
        zenith_angle = np.radians(90.0 - el_deg)
        bphi = g * (np.tan(zenith_angle) + dele)
        # Threw out an (el < 0.01) check here,
        # which will never succeed because el is clipped to be above 1.0 [LS]

        return elevation_rad + np.radians(bphi * sn - aphi)

    @classmethod
    def unrefract(  # noqa: D102 (docstring inherited from base class)
        cls,
        refracted_el_rad,
        pressure_hPa,
        temperature_C,
        humidity_frac,
    ):
        # Maximum difference between input elevation and
        # refraction-corrected version of final output elevation
        tolerance = np.radians(0.01 / 3600)
        # Assume offset from corrected el is similar to offset from uncorrected el
        # -> get lower bound on desired el
        close_offset = (
            cls.refract(refracted_el_rad, pressure_hPa, temperature_C, humidity_frac)
            - refracted_el_rad
        )
        lower = refracted_el_rad - 4 * np.abs(close_offset)
        # We know that corrected el > uncorrected el (mostly)
        # -> this becomes upper bound on desired el
        upper = refracted_el_rad + np.radians(1 / 3600)
        # Do binary search for desired el within this range (but cap iterations
        # in case of a mishap). This assumes that refraction-corrected elevation
        # is monotone function of uncorrected elevation.
        for iteration in range(40):
            el = 0.5 * (lower + upper)
            test_el = cls.refract(el, pressure_hPa, temperature_C, humidity_frac)
            if np.all(np.abs(test_el - refracted_el_rad) < tolerance):
                break
            lower = np.where(test_el < refracted_el_rad, el, lower)
            upper = np.where(test_el > refracted_el_rad, el, upper)
        else:
            logger.warning(
                "Reverse refraction correction did not converge in "
                "%d iterations - elevation differs by at most %f arcsecs",
                iteration + 1,
                np.degrees(np.abs(test_el - refracted_el_rad).max()) * 3600,
            )
        return el if el.ndim else el.item()


# Pick radio frequencies, for which refraction is non-dispersive
_WAVELENGTH_UM = (21 * u.cm).to_value(u.micron)
_CELMIN = 1e-6
_SELMIN = 0.05


def _erfa_refv(el, refa, refb):
    """Source: eraAtioq, from iauAtioq. Very similar to slaRefv."""
    # Cosine and sine of elevation, with precautions
    raet = np.cos(el)
    zaet = np.sin(el)
    r = np.maximum(raet, _CELMIN)
    z = np.maximum(zaet, _SELMIN)
    # A*tan(z)+B*tan^3(z) model, with Newton-Raphson correction
    tz = r / z
    w = refb * tz * tz
    d_el = (refa + w) * tz / (1.0 + (refa + 3.0 * w) / (z * z))
    # Apply the change, giving observed (az, el) vector
    cosdel = 1.0 - d_el * d_el / 2.0
    f = cosdel - d_el * z / r
    raeo = f * raet
    zaeo = cosdel * zaet + d_el * r
    # Observed zenith distance: flip y and x to get observed elevation
    # zdobs = np.arctan2(raeo, zaeo)
    return np.arctan2(zaeo, raeo)


class ErfaRefraction(_RefractionModel):
    """Refraction model used by Astropy, based on ERFA / SOFA.

    Expose the refraction routines used internally by Astropy to transform
    to and from the observed frame, which in turn comes from [ERFA]_.

    Notes
    -----
    The `unrefract` step implements a model dZ = A tan Z + B tan^3 Z, where
    Z is the zenith angle / distance and constants A and B (in radians) are
    derived from the weather parameters by the `erfa.refco` / eraRefco
    routine. This routine in turn traces its lineage to [SOFA]_ and the
    iauRefco routine, and [SLALIB]_ and the sla_REFCOQ routine before that.
    The implementation is a translation of the relevant bit of code found
    in `erfa.atoiq` / eraAtoiq.

    The `refract` step is very similar to the SLALIB sla_REFV routine,
    which inverts the tan(z) model with a single Newton-Raphson iteration.
    This approach first popped up in SOFA release 10 in 2013, which
    introduced the various high-level astrometry routines. This
    implementation follows the code in `erfa.atioq` / eraAtioq, which in
    turn derives from SOFA's iauAtioq.

    References
    ----------
    .. [ERFA] "Essential Routines for Fundamental Astronomy,"
    `<https://github.com/liberfa/erfa>`_
    .. [SOFA] "Standards of Fundamental Astronomy," `<http://www.iausofa.org>`_
    .. [SLALIB] "SLALIB - Positional Astronomy Library," Starlink Project,
    `<https://github.com/scottransom/pyslalib>`_
    """

    @classmethod
    def refract(  # noqa: D102 (docstring inherited from base class)
        cls,
        elevation_rad,
        pressure_hPa,
        temperature_C,
        humidity_frac,
    ):
        # Get constants A and B in model: dZ = A tan Z + B tan^3 Z
        refa, refb = erfa.refco(
            pressure_hPa, temperature_C, humidity_frac, _WAVELENGTH_UM
        )
        # Quick model inversion (1 Newton-Raphson iteration)
        return _erfa_refv(elevation_rad, refa, refb)

    @classmethod
    def unrefract(  # noqa: D102 (docstring inherited from base class)
        cls,
        refracted_el_rad,
        pressure_hPa,
        temperature_C,
        humidity_frac,
    ):
        # Get constants A and B in model: dZ = A tan Z + B tan^3 Z
        refa, refb = erfa.refco(
            pressure_hPa, temperature_C, humidity_frac, _WAVELENGTH_UM
        )
        # Compute refraction at elevation (clipped to avoid blow-up at horizon)
        cos_el = np.cos(refracted_el_rad)
        sin_el = np.maximum(np.sin(refracted_el_rad), _SELMIN)
        tan_z = cos_el / sin_el
        # Model offset is added to zenith angle and therefore subtracted from elevation
        return refracted_el_rad - (refa + refb * tan_z * tan_z) * tan_z


@dataclass(frozen=True)
class TroposphericRefraction:
    """Correct pointing for refractive bending in Earth's atmosphere.

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

    model_id: str = "ErfaRefraction"
    # _model is class itself, not an instance, since all its methods are class methods
    _model: Type[_SomeRefractionModel] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Pick underlying refraction `_model` class based on `model_id`."""
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
        azel: AltAz,
        pressure: u.hPa,
        temperature: u.deg_C,
        relative_humidity: u.dimensionless_unscaled,
    ) -> AltAz:
        """Apply refraction correction to (az, el) coordinates.

        Given `azel`, the direction of incoming electromagnetic waves in the
        absence of an atmosphere, produce `refracted_azel`, the direction from
        which the radiation appears to be coming due to the refractive bending
        induced by the Earth's atmosphere. This therefore transforms space
        coordinates to surface coordinates.

        Parameters
        ----------
        azel : :class:`~astropy.coordinates.AltAz`
            Unrefracted / topocentric / vacuum coordinates
        pressure : :class:`~astropy.units.Quantity`
            Total barometric pressure at surface
        temperature : :class:`~astropy.units.Quantity`
            Ambient air temperature at surface
        relative_humidity : :class:`~astropy.units.Quantity` or float or array
            Relative humidity at surface, as a fraction in range [0, 1]

        Returns
        -------
        refracted_azel : :class:`~astropy.coordinates.AltAz`
            Refracted / observed / surface coordinates
        """
        refracted_el = self._model.refract(
            azel.alt.rad,
            pressure.to_value(u.hPa),
            temperature.to_value(u.deg_C),
            u.Quantity(relative_humidity).to_value(u.dimensionless_unscaled),
        )
        # Clip at zenith, otherwise tiny refraction can go over top and upset Latitude
        refracted_el = np.clip(refracted_el * u.rad, -90 * u.deg, 90 * u.deg)
        data = UnitSphericalRepresentation(azel.az, refracted_el)
        return azel.realize_frame(data)

    @u.quantity_input(equivalencies=u.temperature())
    def unrefract(
        self,
        refracted_azel: AltAz,
        pressure: u.hPa,
        temperature: u.deg_C,
        relative_humidity: u.dimensionless_unscaled,
    ) -> AltAz:
        """Remove refraction correction from (az, el) coordinates.

        Given `refracted_azel`, the observed direction of incoming
        electromagnetic waves in the presence of the Earth's atmosphere,
        produce `azel`, the direction of the radiation in the absence of
        the atmosphere. This therefore transforms surface coordinates to
        space coordinates.

        This undoes a refraction correction that resulted in the given (az, el)
        coordinates. It is the inverse of :meth:`refract`.

        Parameters
        ----------
        refracted_azel : :class:`~astropy.coordinates.AltAz`
            Refracted / observed / surface coordinates
        pressure : :class:`~astropy.units.Quantity`
            Total barometric pressure at surface
        temperature : :class:`~astropy.units.Quantity`
            Ambient air temperature at surface
        relative_humidity : :class:`~astropy.units.Quantity` or float or array
            Relative humidity at surface, as a fraction in range [0, 1]

        Returns
        -------
        azel : :class:`~astropy.coordinates.AltAz`
            Unrefracted / topocentric / vacuum coordinates
        """
        el = self._model.unrefract(
            refracted_azel.alt.rad,
            pressure.to_value(u.hPa),
            temperature.to_value(u.deg_C),
            u.Quantity(relative_humidity).to_value(u.dimensionless_unscaled),
        )
        # Avoid strangeness and clip at -90 degrees
        el = np.clip(el * u.rad, -90 * u.deg, 90 * u.deg)
        data = UnitSphericalRepresentation(refracted_azel.az, el)
        return refracted_azel.realize_frame(data)
