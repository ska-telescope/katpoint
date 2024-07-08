################################################################################
# Copyright (c) 2020-2021,2023-2024, National Research Foundation (SARAO)
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

"""Tropospheric delay model.

This predicts the propagation delay due to neutral gas in the troposphere and
stratosphere as a function of elevation angle, based on the weather, location
and season. The basic model is a direct translation of the model in ALMA's
version of CALC.
"""

from dataclasses import dataclass, field
from typing import Callable

import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation

from ..timestamp import Timestamp

_EXCESS_PATH_PER_PRESSURE = 2.2768 * u.m / u.bar


class SaastamoinenZenithDelay:
    """Zenith delay due to the neutral gas in the troposphere and stratosphere.

    This provides separate methods for the "dry" (hydrostatic) and "wet"
    (non-hydrostatic) components of the atmosphere.

    Parameters
    ----------
    location : :class:`~astropy.coordinates.EarthLocation`
        Location on Earth of observer (used to correct local gravity)

    Notes
    -----
    This is a direct translation of the SASTD and SASTW subroutines in the
    atmospheric module (catmm.f) of Calc 11. It is based on the formulas of
    Saastamoinen [Saas1972]_ as implemented by Davis et al [Davis1985]_. The
    saturation water vapour pressure is calculated by the venerable but still
    practical August-Roche-Magnus formula as discussed in [Murray1967]_.

    References
    ----------
    .. [Saas1972] J. Saastamoinen, “Atmospheric correction for the troposphere
       and stratosphere in radio ranging satellites,” in The Use of Artificial
       Satellites for Geodesy (Geophysical Monograph Series), edited by
       S. W. Henriksen et al, Washington, D.C., vol. 15, pp. 247-251, 1972.
       DOI: 10.1029/GM015p0247

    .. [Davis1985] J. L. Davis, T. A. Herring, I. I. Shapiro, A. E. E. Rogers,
       G. Elgered, “Geodesy by radio interferometry: Effects of atmospheric
       modeling errors on estimates of baseline length,” Radio Science,
       vol. 20, no. 6, pp. 1593-1607, 1985. DOI: 10.1029/rs020i006p01593

    .. [Murray1967] F. W. Murray, “On the computation of saturation vapor
       pressure,” Journal of Applied Meteorology, vol. 6, no. 1, pp. 203-204,
       Feb 1967. DOI: 10.1175/1520-0450(1967)006<0203:OTCOSV>2.0.CO;2
    """

    def __init__(self, location):
        # Reduce local gravity to the value at the centroid of the atmospheric column,
        # which depends on the location of the observer
        self._gravity_correction = (
            1.0 - 0.00266 * np.cos(2 * location.lat) - 0.00028 / u.km * location.height
        )

    @u.quantity_input
    def hydrostatic(self, pressure: u.hPa) -> u.s:
        """Zenith delay due to "dry" (hydrostatic) component of the atmosphere.

        Parameters
        ----------
        pressure : :class:`~astropy.units.Quantity`
            Total barometric pressure at surface

        Returns
        -------
        delay : :class:`~astropy.units.Quantity`
            Zenith delay due to hydrostatic component
        """
        return _EXCESS_PATH_PER_PRESSURE * pressure / self._gravity_correction / const.c

    @u.quantity_input(equivalencies=u.temperature())
    def wet(
        self, temperature: u.deg_C, relative_humidity: u.dimensionless_unscaled
    ) -> u.s:
        """Zenith delay due to "wet" (non-hydrostatic) component of atmosphere.

        Parameters
        ----------
        temperature : :class:`~astropy.units.Quantity`
            Ambient air temperature at surface
        relative_humidity : :class:`~astropy.units.Quantity` or float or array
            Relative humidity at surface, as a fraction in range [0, 1]

        Returns
        -------
        delay : :class:`~astropy.units.Quantity`
            Zenith delay due to non-hydrostatic component
        """
        temp_C = temperature.to_value(u.deg_C, equivalencies=u.temperature())
        temp_K = temperature.to_value(u.K, equivalencies=u.temperature())
        # Get the partial pressure of water vapour for saturated air at given
        # temperature. This resembles the version of the August-Roche-Magnus
        # formula in Murray (1967). The Tetens (1930) version, which serves as
        # the reference, is saturation_pressure_hPa =
        # 10 ** (7.5 * temperature_C / (temperature_C + 237.3) + 0.7858).
        saturation_pressure = 6.11 * np.exp(17.269 * temp_C / (temp_C + 237.3)) * u.hPa
        partial_pressure = relative_humidity * saturation_pressure
        excess_path_per_pressure = _EXCESS_PATH_PER_PRESSURE * (1255.0 / temp_K + 0.05)
        return excess_path_per_pressure * partial_pressure / const.c


_ZENITH_DELAY = {"SaastamoinenZenithDelay": SaastamoinenZenithDelay}

_GMF_H_MEAN_COEFS = np.array(
    [
        +1.2517e02,
        +8.503e-01,
        +6.936e-02,
        -6.760e00,
        +1.771e-01,  # ah_mean
        +1.130e-02,
        +5.963e-01,
        +1.808e-02,
        +2.801e-03,
        -1.414e-03,
        -1.212e00,
        +9.300e-02,
        +3.683e-03,
        +1.095e-03,
        +4.671e-05,
        +3.959e-01,
        -3.867e-02,
        +5.413e-03,
        -5.289e-04,
        +3.229e-04,
        +2.067e-05,
        +3.000e-01,
        +2.031e-02,
        +5.900e-03,
        +4.573e-04,
        -7.619e-05,
        +2.327e-06,
        +3.845e-06,
        +1.182e-01,
        +1.158e-02,
        +5.445e-03,
        +6.219e-05,
        +4.204e-06,
        -2.093e-06,
        +1.540e-07,
        -4.280e-08,
        -4.751e-01,
        -3.490e-02,
        +1.758e-03,
        +4.019e-04,
        -2.799e-06,
        -1.287e-06,
        +5.468e-07,
        +7.580e-08,
        -6.300e-09,
        -1.160e-01,
        +8.301e-03,
        +8.771e-04,
        +9.955e-05,
        -1.718e-06,
        -2.012e-06,
        +1.170e-08,
        +1.790e-08,
        -1.300e-09,
        +1.000e-10,
        +0.000e00,
        +0.000e00,
        +3.249e-02,
        +0.000e00,
        +3.324e-02,  # bh_mean
        +1.850e-02,
        +0.000e00,
        -1.115e-01,
        +2.519e-02,
        +4.923e-03,
        +0.000e00,
        +2.737e-02,
        +1.595e-02,
        -7.332e-04,
        +1.933e-04,
        +0.000e00,
        -4.796e-02,
        +6.381e-03,
        -1.599e-04,
        -3.685e-04,
        +1.815e-05,
        +0.000e00,
        +7.033e-02,
        +2.426e-03,
        -1.111e-03,
        -1.357e-04,
        -7.828e-06,
        +2.547e-06,
        +0.000e00,
        +5.779e-03,
        +3.133e-03,
        -5.312e-04,
        -2.028e-05,
        +2.323e-07,
        -9.100e-08,
        -1.650e-08,
        +0.000e00,
        +3.688e-02,
        -8.638e-04,
        -8.514e-05,
        -2.828e-05,
        +5.403e-07,
        +4.390e-07,
        +1.350e-08,
        +1.800e-09,
        +0.000e00,
        -2.736e-02,
        -2.977e-04,
        +8.113e-05,
        +2.329e-07,
        +8.451e-07,
        +4.490e-08,
        -8.100e-09,
        -1.500e-09,
        +2.000e-10,
    ]
)

_GMF_H_AMPLITUDE_COEFS = np.array(
    [
        -2.738e-01,
        -2.837e00,
        +1.298e-02,
        -3.588e-01,
        +2.413e-02,  # ah_amp
        +3.427e-02,
        -7.624e-01,
        +7.272e-02,
        +2.160e-02,
        -3.385e-03,
        +4.424e-01,
        +3.722e-02,
        +2.195e-02,
        -1.503e-03,
        +2.426e-04,
        +3.013e-01,
        +5.762e-02,
        +1.019e-02,
        -4.476e-04,
        +6.790e-05,
        +3.227e-05,
        +3.123e-01,
        -3.535e-02,
        +4.840e-03,
        +3.025e-06,
        -4.363e-05,
        +2.854e-07,
        -1.286e-06,
        -6.725e-01,
        -3.730e-02,
        +8.964e-04,
        +1.399e-04,
        -3.990e-06,
        +7.431e-06,
        -2.796e-07,
        -1.601e-07,
        +4.068e-02,
        -1.352e-02,
        +7.282e-04,
        +9.594e-05,
        +2.070e-06,
        -9.620e-08,
        -2.742e-07,
        -6.370e-08,
        -6.300e-09,
        +8.625e-02,
        -5.971e-03,
        +4.705e-04,
        +2.335e-05,
        +4.226e-06,
        +2.475e-07,
        -8.850e-08,
        -3.600e-08,
        -2.900e-09,
        +0.000e00,
        +0.000e00,
        +0.000e00,
        -1.136e-01,
        +0.000e00,
        -1.868e-01,  # bh_amp
        -1.399e-02,
        +0.000e00,
        -1.043e-01,
        +1.175e-02,
        -2.240e-03,
        +0.000e00,
        -3.222e-02,
        +1.333e-02,
        -2.647e-03,
        -2.316e-05,
        +0.000e00,
        +5.339e-02,
        +1.107e-02,
        -3.116e-03,
        -1.079e-04,
        -1.299e-05,
        +0.000e00,
        +4.861e-03,
        +8.891e-03,
        -6.448e-04,
        -1.279e-05,
        +6.358e-06,
        -1.417e-07,
        +0.000e00,
        +3.041e-02,
        +1.150e-03,
        -8.743e-04,
        -2.781e-05,
        +6.367e-07,
        -1.140e-08,
        -4.200e-08,
        +0.000e00,
        -2.982e-02,
        -3.000e-03,
        +1.394e-05,
        -3.290e-05,
        -1.705e-07,
        +7.440e-08,
        +2.720e-08,
        -6.600e-09,
        +0.000e00,
        +1.236e-02,
        -9.981e-04,
        -3.792e-05,
        -1.355e-05,
        +1.162e-06,
        -1.789e-07,
        +1.470e-08,
        -2.400e-09,
        -4.000e-10,
    ]
)

_GMF_W_MEAN_COEFS = np.array(
    [
        +5.640e01,
        +1.555e00,
        -1.011e00,
        -3.975e00,
        +3.171e-02,  # aw_mean
        +1.065e-01,
        +6.175e-01,
        +1.376e-01,
        +4.229e-02,
        +3.028e-03,
        +1.688e00,
        -1.692e-01,
        +5.478e-02,
        +2.473e-02,
        +6.059e-04,
        +2.278e00,
        +6.614e-03,
        -3.505e-04,
        -6.697e-03,
        +8.402e-04,
        +7.033e-04,
        -3.236e00,
        +2.184e-01,
        -4.611e-02,
        -1.613e-02,
        -1.604e-03,
        +5.420e-05,
        +7.922e-05,
        -2.711e-01,
        -4.406e-01,
        -3.376e-02,
        -2.801e-03,
        -4.090e-04,
        -2.056e-05,
        +6.894e-06,
        +2.317e-06,
        +1.941e00,
        -2.562e-01,
        +1.598e-02,
        +5.449e-03,
        +3.544e-04,
        +1.148e-05,
        +7.503e-06,
        -5.667e-07,
        -3.660e-08,
        +8.683e-01,
        -5.931e-02,
        -1.864e-03,
        -1.277e-04,
        +2.029e-04,
        +1.269e-05,
        +1.629e-06,
        +9.660e-08,
        -1.015e-07,
        -5.000e-10,
        +0.000e00,
        +0.000e00,
        +2.592e-01,
        +0.000e00,
        +2.974e-02,  # bw_mean
        -5.471e-01,
        +0.000e00,
        -5.926e-01,
        -1.030e-01,
        -1.567e-02,
        +0.000e00,
        +1.710e-01,
        +9.025e-02,
        +2.689e-02,
        +2.243e-03,
        +0.000e00,
        +3.439e-01,
        +2.402e-02,
        +5.410e-03,
        +1.601e-03,
        +9.669e-05,
        +0.000e00,
        +9.502e-02,
        -3.063e-02,
        -1.055e-03,
        -1.067e-04,
        -1.130e-04,
        +2.124e-05,
        +0.000e00,
        -3.129e-01,
        +8.463e-03,
        +2.253e-04,
        +7.413e-05,
        -9.376e-05,
        -1.606e-06,
        +2.060e-06,
        +0.000e00,
        +2.739e-01,
        +1.167e-03,
        -2.246e-05,
        -1.287e-04,
        -2.438e-05,
        -7.561e-07,
        +1.158e-06,
        +4.950e-08,
        +0.000e00,
        -1.344e-01,
        +5.342e-03,
        +3.775e-04,
        -6.756e-05,
        -1.686e-06,
        -1.184e-06,
        +2.768e-07,
        +2.730e-08,
        +5.700e-09,
    ]
)

_GMF_W_AMPLITUDE_COEFS = np.array(
    [
        +1.023e-01,
        -2.695e00,
        +3.417e-01,
        -1.405e-01,
        +3.175e-01,  # aw_amp
        +2.116e-01,
        +3.536e00,
        -1.505e-01,
        -1.660e-02,
        +2.967e-02,
        +3.819e-01,
        -1.695e-01,
        -7.444e-02,
        +7.409e-03,
        -6.262e-03,
        -1.836e00,
        -1.759e-02,
        -6.256e-02,
        -2.371e-03,
        +7.947e-04,
        +1.501e-04,
        -8.603e-01,
        -1.360e-01,
        -3.629e-02,
        -3.706e-03,
        -2.976e-04,
        +1.857e-05,
        +3.021e-05,
        +2.248e00,
        -1.178e-01,
        +1.255e-02,
        +1.134e-03,
        -2.161e-04,
        -5.817e-06,
        +8.836e-07,
        -1.769e-07,
        +7.313e-01,
        -1.188e-01,
        +1.145e-02,
        +1.011e-03,
        +1.083e-04,
        +2.570e-06,
        -2.140e-06,
        -5.710e-08,
        +2.000e-08,
        -1.632e00,
        -6.948e-03,
        -3.893e-03,
        +8.592e-04,
        +7.577e-05,
        +4.539e-06,
        -3.852e-07,
        -2.213e-07,
        -1.370e-08,
        +5.800e-09,
        +0.000e00,
        +0.000e00,
        -8.865e-02,
        +0.000e00,
        -4.309e-01,  # bw_amp
        +6.340e-02,
        +0.000e00,
        +1.162e-01,
        +6.176e-02,
        -4.234e-03,
        +0.000e00,
        +2.530e-01,
        +4.017e-02,
        -6.204e-03,
        +4.977e-03,
        +0.000e00,
        -1.737e-01,
        -5.638e-03,
        +1.488e-04,
        +4.857e-04,
        -1.809e-04,
        +0.000e00,
        -1.514e-01,
        -1.685e-02,
        +5.333e-03,
        -7.611e-05,
        +2.394e-05,
        +8.195e-06,
        +0.000e00,
        +9.326e-02,
        -1.275e-02,
        -3.071e-04,
        +5.374e-05,
        -3.391e-05,
        -7.436e-06,
        +6.747e-07,
        +0.000e00,
        -8.637e-02,
        -3.807e-03,
        -6.833e-04,
        -3.861e-05,
        -2.268e-05,
        +1.454e-06,
        +3.860e-07,
        -1.068e-07,
        +0.000e00,
        -2.658e-02,
        -1.947e-03,
        +7.131e-04,
        -3.506e-05,
        +1.885e-07,
        +5.792e-07,
        +3.990e-08,
        +2.000e-08,
        -5.700e-09,
    ]
)


def _associated_legendre_polynomials(n, m, x):
    """Matrix of associated Legendre polynomials.

    This computes the associated Legendre polynomials :math:`P_n^m(x)` of
    degrees ``0..n`` and orders ``0..m``, evaluated at the real value `x`.

    Parameters
    ----------
    n, m : int
        Maximum degree and order, respectively, of Legendre polynomials
    x : float
        Polynomial input value, typically in interval (-1, 1)

    Returns
    -------
    Pnm_x : array of float, shape (n+1, m+1)
        Values for all degrees 0..n and orders 0..m

    Notes
    -----
    This was translated from the GMF11 subroutine in the atmospheric module
    (catmm.f) of Calc 11. It is based on the formula in [HM1967]_, which seems
    accurate as long as n and m are not too large. It does not include the
    Condon–Shortley phase term (:math:`(-1)^m`). It closely matches the output
    of the following SciPy code::

        scipy.special.lpmn(m, n, x)[0].T * (-1) ** np.arange(m + 1)

    References
    ----------
    .. [HM1967] V.A. Heiskanen, H. Moritz, "Physical Geodesy," W.H. Freeman and
       Company, San Francisco, London, 1967, ISBN 978-0716702337, Equation 1-62.
    """
    # Sequence of factorials from 0..2n + 1
    fact = np.ones(2 * n + 2)
    fact[2:] = np.cumprod(np.arange(2.0, len(fact)))
    P = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        for j in range(min(i, m) + 1):
            ir = (i - j) // 2
            s = 0
            for k in range(ir + 1):
                s += (
                    (-1) ** k
                    * fact[2 * i - 2 * k]
                    / fact[k]
                    / fact[i - k]
                    / fact[i - j - 2 * k]
                    * x ** (i - j - 2 * k)
                )
            P[i, j] = 1.0 / 2**i * np.sqrt((1 - x**2) ** j) * s
    return P


def _continued_fraction(elevation, a, b, c):
    """Marini-style continued fraction evaluated at given elevation angle."""
    # Express formula in terms of zenith angle z to match notation in references
    cos_z = np.sin(elevation)
    topcon = 1.0 + a / (1.0 + b / (1.0 + c))
    return topcon / (cos_z + a / (cos_z + b / (cos_z + c)))


def _niell_season(timestamp):
    """Seasonal sinusoidal variation as used in Niell's mapping function."""
    time = Timestamp(timestamp).time
    # Subtract the first day of an arbitrary year (1980) to line up the phase.
    # Reference day is 28 January, consistent with Niell (1996).
    day_of_year = time.utc.mjd - 44239 + 1 - 28
    # Middle of Northern hemisphere winter = +1, middle of summer = -1
    return np.cos(2.0 * np.pi * day_of_year / 365.25)


class GlobalMappingFunction:
    """Function that describes the elevation dependence of atmospheric delay.

    This maps zenith delays to any elevation angle, based on the site coordinates
    and the day of year. It provides separate methods for the "dry" (hydrostatic)
    and "wet" (non-hydrostatic) components of the troposphere and stratosphere.

    Parameters
    ----------
    location : :class:`~astropy.coordinates.EarthLocation`
        Location on Earth of observer (used in global weather model)

    Notes
    -----
    This is a direct translation of the GMF11 subroutine in the atmospheric
    module (catmm.f) of Calc 11, which in turn is based on Fortran code
    associated with [Boehm2006]_. This paper describes the Global Mapping
    Function (GMF), a static model describing average weather conditions
    as a function of latitude, longitude, height and the day of year, which
    was fit to three years of global weather data. This is a refinement of
    Niell's mapping function [Niell1996]_ and shares some of its formulas.

    References
    ----------
    .. [Boehm2006] J. Boehm, A. Niell, P. Tregoning, H. Schuh, “Global Mapping
       Function (GMF): A new empirical mapping function based on numerical
       weather model data,” Geophysical Research Letters, vol. 33, no. L07304,
       Apr 2006. DOI: 10.1029/2005GL025546

    .. [Niell1996] A.E. Niell, “Global mapping functions for the atmosphere delay
       at radio wavelengths,” Journal of Geophysical Research: Solid Earth, vol.
       101, no. B2, pp. 3227–3246, Feb 1996. DOI: 10.1029/95JB03048
    """

    def __init__(self, location):
        self.location = location
        # Obtain 10x10 matrix of Legendre values evaluated at cos(colatitude)
        P = _associated_legendre_polynomials(n=9, m=9, x=np.sin(location.lat))
        m_longitude = np.arange(P.shape[1]) * location.lon
        # aP and bP are related to the output of SciPy's spherical harmonic function.
        # Given y_mn = scipy.special.sph_harm(m, n, theta, phi):
        #  - m and n have the same meaning (order and degree)
        #  - theta -> longitude, phi -> colatitude (90 degrees - latitude)
        #  - aP / bP -> real / imag part of y_mn,
        #    without sqrt scale factor and for all m and n
        aP = P * np.cos(m_longitude)
        bP = P * np.sin(m_longitude)
        tril = np.tril_indices_from(P)
        # Unravel matrices into basis vector of length 110,
        # include coef scale factor of 1e-5.
        self._spherical_harmonic_basis = 1e-5 * np.r_[aP[tril], bP[tril]]

    @u.quantity_input
    def hydrostatic(self, elevation: u.rad, timestamp) -> u.dimensionless_unscaled:
        """Perform mapping for "dry" (hydrostatic) component of the atmosphere.

        Parameters
        ----------
        elevation : :class:`~astropy.units.Quantity` or
            :class:`~astropy.coordinates.Angle`
            Elevation angle
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equivalent
            Observation time (to incorporate seasonal weather patterns)

        Returns
        -------
        gmf : :class:`~astropy.units.Quantity`
            Scale factor that turns zenith delay into delay at elevation angle
        """
        latitude = self.location.lat
        height = self.location.height
        a_mean = _GMF_H_MEAN_COEFS @ self._spherical_harmonic_basis
        a_amplitude = _GMF_H_AMPLITUDE_COEFS @ self._spherical_harmonic_basis
        season = _niell_season(timestamp)
        a = a_mean + a_amplitude * season
        b = 0.0029
        c0 = 0.062
        if latitude < 0:
            # Southern hemisphere has the opposite season
            # We flip the season for c but not a, since a already has a global model
            season = -season
            c11 = 0.007
            c10 = 0.002
        else:
            # Northern hemisphere
            c11 = 0.005
            c10 = 0.001
        c = c0 + ((season + 1) * c11 / 2 + c10) * (1 - np.cos(latitude))
        gmf = _continued_fraction(elevation, a, b, c)
        # Niell's hydrostatic height correction (<0.05% for 1 km altitude)
        correction = 1.0 / np.sin(elevation) - _continued_fraction(
            elevation, a=2.53e-5, b=5.49e-3, c=1.14e-3
        )
        return gmf + correction / u.km * height

    @u.quantity_input
    def wet(self, elevation: u.rad, timestamp) -> u.dimensionless_unscaled:
        """Perform mapping for "wet" (non-hydrostatic) component of the atmosphere.

        Parameters
        ----------
        elevation : :class:`~astropy.units.Quantity` or
            :class:`~astropy.coordinates.Angle`
            Elevation angle
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equivalent
            Observation time (to incorporate seasonal weather patterns)

        Returns
        -------
        gmf : :class:`~astropy.units.Quantity`
            Scale factor that turns zenith delay into delay at elevation angle
        """
        a_mean = _GMF_W_MEAN_COEFS @ self._spherical_harmonic_basis
        a_amplitude = _GMF_W_AMPLITUDE_COEFS @ self._spherical_harmonic_basis
        season = _niell_season(timestamp)
        a = a_mean + a_amplitude * season
        b = 0.00146
        c = 0.04391
        return _continued_fraction(elevation, a, b, c)


_MAPPING_FUNCTION = {"GlobalMappingFunction": GlobalMappingFunction}


@dataclass(frozen=True)
class TroposphericDelay:
    """Propagation delay due to neutral gas in the troposphere and stratosphere.

    Set up a tropospheric delay model as specified by the model ID with format::

         "<zenith delay>-<mapping function>[-<hydrostatic/wet>]"

    This picks an appropriate zenith delay formula and mapping function, and
    optionally restricts the delays to hydrostatic or wet components only.
    The delays are calculated by calling this object like a function.

    Parameters
    ----------
    location : :class:`~astropy.coordinates.EarthLocation`
        Location on Earth of observer
    model_id : str, optional
        Unique identifier of tropospheric model (defaults to the only model
        implemented so far)

    Raises
    ------
    ValueError
        If the specified tropospheric model is unknown or has wrong format
    """

    location: EarthLocation
    model_id: str = "SaastamoinenZenithDelay-GlobalMappingFunction"
    _delay: Callable = field(init=False, repr=False, compare=False)
    # EarthLocation is not hashable, but dataclass assumes it is, so disable it
    __hash__ = None

    def __post_init__(self):
        """Initialise main function `_delay` from `location` and `model_id`."""
        # Parse model identifier string
        model_parts = self.model_id.split("-")
        if len(model_parts) == 2:
            model_parts.append("total")
        if len(model_parts) != 3:
            raise ValueError(
                f"Format for tropospheric delay model ID is '<zenith delay>-"
                f"<mapping function>[-<hydrostatic/wet>]', not {self.model_id!r}"
            )

        def get(mapping, key, name):
            try:
                return mapping[key]
            except KeyError as err:
                raise ValueError(
                    f"Tropospheric delay model {self.model_id!r} has unknown {name} "
                    f"{key!r}, available ones are {list(mapping.keys())}"
                ) from err

        zenith_delay = get(_ZENITH_DELAY, model_parts[0], "zenith delay function")(
            self.location
        )
        mapping_function = get(_MAPPING_FUNCTION, model_parts[1], "mapping function")(
            self.location
        )

        def hydrostatic(p, t, h, el, ts):  # pylint: disable=unused-argument
            return zenith_delay.hydrostatic(p) * mapping_function.hydrostatic(el, ts)

        def wet(p, t, h, el, ts):  # pylint: disable=unused-argument
            return zenith_delay.wet(t, h) * mapping_function.wet(el, ts)

        def total(p, t, h, el, ts):
            return hydrostatic(p, t, h, el, ts) + wet(p, t, h, el, ts)

        model_types = {"hydrostatic": hydrostatic, "wet": wet, "total": total}
        # Set attribute on base class because this class is frozen
        super().__setattr__("_delay", get(model_types, model_parts[2], "type"))

    @u.quantity_input(equivalencies=u.temperature())
    def __call__(
        self,
        pressure: u.hPa,
        temperature: u.deg_C,
        relative_humidity: u.dimensionless_unscaled,
        elevation: u.rad,
        timestamp,
    ) -> u.s:
        """Propagation delay due to neutral gas in the troposphere and stratosphere.

        Parameters
        ----------
        pressure : :class:`~astropy.units.Quantity`
            Total barometric pressure at surface
        temperature : :class:`~astropy.units.Quantity`
            Ambient air temperature at surface
        relative_humidity : :class:`~astropy.units.Quantity` or float or array
            Relative humidity at surface, as a fraction in range [0, 1]
        elevation : :class:`~astropy.units.Quantity` or
            :class:`~astropy.coordinates.Angle`
            Elevation angle
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equivalent
            Observation time (to incorporate seasonal weather patterns)

        Returns
        -------
        delay : :class:`~astropy.units.Quantity`
            Tropospheric propagation delay
        """
        return self._delay(
            pressure,
            temperature,
            relative_humidity,
            elevation,
            timestamp,
        )
