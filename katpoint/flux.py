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

"""Flux density model."""

import warnings

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
    min_freq_MHz : float or string
        Minimum frequency for which model is valid, in MHz. Alternatively, this
        is a description string containing the minimum frequency, maximum
        frequency and model coefficients as space-separated values (optionally
        with parentheses enclosing the entire string).
    max_freq_MHz : float, optional
        Maximum frequency for which model is valid, in MHz
    coefs : sequence of floats, optional
        Model coefficients (a, b, c, d, e, f, I, Q, U, V), where missing
        coefficients at the end of the sequence are assumed to be zero (except
        for I, assumes to be one), and extra coefficients are ignored.

    Raises
    ------
    ValueError
        If description string has the wrong format or is mixed with normal
        parameters

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
    _DEFAULT_COEFS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,    # a, b, c, d, e, f
                               1.0, 0.0, 0.0, 0.0])             # I, Q, U, V

    def __init__(self, min_freq_MHz, max_freq_MHz=None, coefs=None):
        # If the first parameter is a description string, extract the relevant flux parameters from it
        if isinstance(min_freq_MHz, str):
            # Cannot have other parameters if description string is given - this is a safety check
            if not (max_freq_MHz is None and coefs is None):
                raise ValueError(f"First parameter '{min_freq_MHz}' is description string - "
                                 "cannot have other parameters")
            # Split description string on spaces and turn into numbers (discarding any parentheses)
            try:
                flux_info = [float(num) for num in min_freq_MHz.strip(' ()').split()]
            except ValueError as err:
                raise FluxError(f"Floating point number '{min_freq_MHz}' is invalid") from err
            if len(flux_info) < 2:
                raise FluxError(f"Flux density description string '{min_freq_MHz}' is invalid")
            min_freq_MHz, max_freq_MHz, coefs = flux_info[0], flux_info[1], tuple(flux_info[2:])
        self.min_freq_MHz = min_freq_MHz
        self.max_freq_MHz = max_freq_MHz
        self.coefs = self._DEFAULT_COEFS.copy()
        # Extract up to the maximum number of coefficients from given sequence
        if len(coefs) > len(self.coefs):
            warnings.warn(f'Received {len(coefs)} coefficients but only expected {len(self.coefs)} - '
                          'ignoring the rest', FutureWarning)
        self.coefs[:min(len(self.coefs), len(coefs))] = coefs[:min(len(self.coefs), len(coefs))]
        # Prune defaults at the end of coefficient list for the description string
        nondefault_coefs = np.nonzero(self.coefs != self._DEFAULT_COEFS)[0]
        last_nondefault_coef = nondefault_coefs[-1] if len(nondefault_coefs) > 0 else 0
        pruned_coefs = self.coefs[:last_nondefault_coef + 1]
        coefs_str = ' '.join([repr(c) for c in pruned_coefs])
        self.description = f'({min_freq_MHz} {max_freq_MHz} {coefs_str})'

    def __str__(self):
        """Verbose human-friendly string representation."""
        freq_range = f'{self.min_freq_MHz:.0f}-{self.max_freq_MHz:.0f} MHz'
        coefs_str = ', '.join([repr(c) for c in self.coefs])
        return f"Flux density defined for {freq_range}, coefs=({coefs_str})"

    def __repr__(self):
        """Short human-friendly string representation."""
        freq_range = f'{self.min_freq_MHz:.0f}-{self.max_freq_MHz:.0f} MHz'
        param_str = ','.join(np.array(list('abcdefIQUV'))[self.coefs != self._DEFAULT_COEFS])
        return f"<katpoint.FluxDensityModel {freq_range} params={param_str} at {id(self):#x}>"

    def __eq__(self, other):
        """Equality comparison operator (based on description string)."""
        return self.description == \
            (other.description if isinstance(other, self.__class__) else other)

    def __hash__(self):
        """Base hash on description string, just like equality operator."""
        return hash(self.description)

    @property
    def iquv_scale(self):
        """Fractional Stokes parameters which scale the flux density."""
        return self.coefs[6:10]

    def _flux_density_raw(self, freq_MHz):
        a, b, c, d, e, f = self.coefs[:6]
        log10_v = np.log10(freq_MHz)
        log10_S = a + b * log10_v + c * log10_v ** 2 + d * log10_v ** 3 + e * np.exp(f * log10_v)
        return 10 ** log10_S

    def flux_density(self, freq_MHz):
        """Calculate Stokes I flux density for given observation frequency.

        Parameters
        ----------
        freq_MHz : float, or sequence of floats
            Frequency at which to evaluate flux density, in MHz

        Returns
        -------
        flux_density : float, or array of floats of same shape as *freq_MHz*
            Flux density in Jy, or np.nan if the frequency is out of range
        """
        freq_MHz = np.asarray(freq_MHz)
        flux = np.asarray(self._flux_density_raw(freq_MHz) * self.iquv_scale[0])
        flux[freq_MHz < self.min_freq_MHz] = np.nan
        flux[freq_MHz > self.max_freq_MHz] = np.nan
        return flux if flux.ndim else flux.item()

    def flux_density_stokes(self, freq_MHz):
        """Calculate full-Stokes flux density for given observation frequency.

        Parameters
        ----------
        freq_MHz : float, or sequence of floats
            Frequency at which to evaluate flux density, in MHz

        Returns
        -------
        flux_density : array of floats
            Flux density in Jy, or np.nan if the frequency is out of range. The
            array has an extra final axis of length 4, corresponding to the I, Q, U, V
            components.
        """
        freq_MHz = np.asarray(freq_MHz)
        flux = np.asarray(self._flux_density_raw(freq_MHz))
        flux[freq_MHz < self.min_freq_MHz] = np.nan
        flux[freq_MHz > self.max_freq_MHz] = np.nan
        return np.multiply.outer(flux, self.iquv_scale)
