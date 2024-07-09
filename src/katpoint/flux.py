################################################################################
# Copyright (c) 2009-2011,2013,2016-2021,2023, National Research Foundation (SARAO)
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

"""Flux density model."""

import warnings

import astropy.units as u
import numpy as np


class FluxError(ValueError):
    """Exception for a flux parsing error."""


class FluxDensityModel:
    """Spectral flux density model.

    This models the spectral flux density (or spectral energy distribtion - SED)
    of a radio source as::

       log10(S) = a + b*log10(v) + c*log10(v)**2 + d*log10(v)**3 + e*exp(f*log10(v))

    where *S* is the flux density in janskies (Jy) and *v* is the frequency in
    MHz. The model is based on the Baars polynomial [BGP1977]_ (up to a third-
    order term) and extended with an exponential term from the 1Jy catalogue
    [KWP+1981]_. It is considered valid for a specified frequency range only.
    For any frequencies outside this range a value of NaN is returned.

    It also models polarisation: an optional (I, Q, U, V) vector may be given
    to specify fractional Stokes parameters, which scale *S*. If not specified,
    the default is unpolarised (I = 1, Q = U = V = 0). It is recommended that I
    is left at 1, but it can be changed to model non-physical sources e.g.
    negative CLEAN components.

    The object can be instantiated directly with the minimum and maximum
    frequencies of the valid frequency range and the model coefficients, or
    indirectly via a description string. This string contains the minimum
    frequency, maximum frequency and model coefficients as space-separated values
    (optionally with parentheses enclosing the entire string). Some examples::

       '1000.0 2000.0 0.34 -0.85 -0.02'
       '(1000.0 2000.0 0.34 -0.85 0.0 0.0 2.3 -1.0)'
       '1000.0 2000.0 0.34 -0.85 0.0 0.0 2.3 -1.0  1.0 0.2 -0.1 0.0'

    If less than the expected number of coefficients are provided, the rest are
    assumed to be zero, except that *I* is assumed to be one. If more than the
    expected number are provided, the extra coefficients are ignored, but a
    warning is shown.

    Parameters
    ----------
    min_frequency, max_frequency : :class:`~astropy.units.Quantity`
        Minimum and maximum frequency for which model is valid
    coefs : sequence of floats, optional
        Model coefficients (a, b, c, d, e, f, I, Q, U, V), where missing
        coefficients at the end of the sequence are assumed to be zero
        (except for I, assumed to be one), and extra coefficients are ignored.

    References
    ----------
    .. [BGP1977] J.W.M. Baars, R. Genzel, I.I.K. Pauliny-Toth, A. Witzel,
       "The Absolute Spectrum of Cas A; An Accurate Flux Density Scale and
       a Set of Secondary Calibrators," Astron. Astrophys., 61, 99-106, 1977.
    .. [KWP+1981] H. Kuehr, A. Witzel, I.I.K. Pauliny-Toth, U. Nauber,
       "A catalogue of extragalactic radio sources having flux densities greater
       than 1 Jy at 5 GHz," Astron. Astrophys. Suppl. Ser., 45, 367-430, 1981.
    """

    # Coefficients are zero by default, except for I
    _DEFAULT_COEFS = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # a, b, c, d, e, f
    )  # I, Q, U, V

    @u.quantity_input(equivalencies=u.spectral())
    def __init__(self, min_frequency: u.Hz, max_frequency: u.Hz, coefs):
        self.min_frequency = min_frequency << u.MHz
        self.max_frequency = max_frequency << u.MHz
        self.coefs = self._DEFAULT_COEFS.copy()
        # Extract up to the maximum number of coefficients from given sequence
        if len(coefs) > len(self.coefs):
            warnings.warn(
                f"Received {len(coefs)} coefficients but only expected "
                f"{len(self.coefs)} - ignoring the rest",
                FutureWarning,
            )
        self.coefs[: min(len(self.coefs), len(coefs))] = coefs[
            : min(len(self.coefs), len(coefs))
        ]

    def __str__(self):
        """Complete string representation of object, sufficient to reconstruct it."""
        return self.description

    def __repr__(self):
        """Short human-friendly string representation."""
        min_freq = self.min_frequency.to_value(u.MHz)
        max_freq = self.max_frequency.to_value(u.MHz)
        freq_range = f"{min_freq:.0f}-{max_freq:.0f} MHz"
        param_str = ",".join(
            np.array(list("abcdefIQUV"))[self.coefs != self._DEFAULT_COEFS]
        )
        return (
            f"<katpoint.FluxDensityModel {freq_range} params={param_str} "
            f"at {id(self):#x}>"
        )

    def __eq__(self, other):
        """Equality comparison operator (based on description string)."""
        return self.description == (
            other.description if isinstance(other, self.__class__) else other
        )

    def __hash__(self):
        """Compute hash on description string, just like equality operator."""
        return hash(self.description)

    @property
    def description(self):
        """Complete string representation of object, sufficient to reconstruct it."""
        min_freq = self.min_frequency.to_value(u.MHz)
        max_freq = self.max_frequency.to_value(u.MHz)
        # Prune defaults at the end of coefficient list for the description string
        nondefault_coefs = np.nonzero(self.coefs != self._DEFAULT_COEFS)[0]
        last_nondefault_coef = nondefault_coefs[-1] if len(nondefault_coefs) > 0 else 0
        pruned_coefs = self.coefs[: last_nondefault_coef + 1]
        coefs_str = " ".join([repr(c) for c in pruned_coefs])
        return f"({min_freq} {max_freq} {coefs_str})"

    @classmethod
    def from_description(cls, description):
        """Construct flux density model object from description string.

        Parameters
        ----------
        description : str
            String of space-separated parameters (optionally in parentheses)

        Returns
        -------
        flux_model : :class:`FluxDensityModel`
            Constructed flux density model object

        Raises
        ------
        FluxError
            If `description` has the wrong format
        """
        # Split description string on spaces and turn into numbers
        # (discarding any parentheses).
        prefix = f"Flux density description string '{description}'"
        try:
            flux_info = [float(num) for num in description.strip(" ()").split()]
        except ValueError as err:
            raise FluxError(f"{prefix} contains invalid floats") from err
        if len(flux_info) < 2:
            raise FluxError(f"{prefix} should have at least two parameters")
        return cls(flux_info[0] * u.MHz, flux_info[1] * u.MHz, flux_info[2:])

    @property
    def iquv_scale(self):
        """Fractional Stokes parameters which scale the flux density."""
        return self.coefs[6:10]

    @u.quantity_input
    def _flux_density_raw(self, frequency: u.Hz) -> u.Jy:
        a, b, c, d, e, f = self.coefs[:6]
        log10_v = np.log10(frequency.to_value(u.MHz))
        log10_S = (
            a + b * log10_v + c * log10_v**2 + d * log10_v**3 + e * np.exp(f * log10_v)
        )
        return 10**log10_S * u.Jy

    @u.quantity_input(equivalencies=u.spectral())
    def flux_density(self, frequency: u.Hz) -> u.Jy:
        """Calculate Stokes I flux density for given observation frequency.

        Parameters
        ----------
        frequency : :class:`~astropy.units.Quantity`, optional
            Frequency at which to evaluate flux density

        Returns
        -------
        flux_density : :class:`~astropy.units.Quantity`
            Flux density, or NaN Jy if frequency is out of range.
            The shape matches the input.
        """
        frequency <<= u.MHz
        flux = self._flux_density_raw(frequency) * self.iquv_scale[0]
        flux[frequency < self.min_frequency] = np.nan * u.Jy
        flux[frequency > self.max_frequency] = np.nan * u.Jy
        return flux

    @u.quantity_input(equivalencies=u.spectral())
    def flux_density_stokes(self, frequency: u.Hz) -> u.Jy:
        """Calculate full-Stokes flux density for given observation frequency.

        Parameters
        ----------
        frequency : :class:`~astropy.units.Quantity`, optional
            Frequency at which to evaluate flux density

        Returns
        -------
        flux_density : :class:`~astropy.units.Quantity`
            Flux density, or NaN Jy if frequency is out of range.
            The shape matches the input with an extra trailing dimension
            of size 4 containing Stokes I, Q, U, V.
        """
        frequency <<= u.MHz
        flux = self._flux_density_raw(frequency)
        flux[frequency < self.min_frequency] = np.nan * u.Jy
        flux[frequency > self.max_frequency] = np.nan * u.Jy
        return np.multiply.outer(flux, self.iquv_scale)
