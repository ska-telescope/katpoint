################################################################################
# Copyright (c) 2009-2010,2016,2018-2023, National Research Foundation (SARAO)
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

"""Geodetic and spherical coordinate transformations, and angle conversions."""

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle

# -------------------------------------------------------------------------------------
# --- Angle conversion utilities
# -------------------------------------------------------------------------------------


def to_angle(s, sexagesimal_unit=u.deg):
    """Construct an `Angle` with default units.

    This creates an :class:`~astropy.coordinates.Angle` with the following
    default units:

      - A number is in radians.
      - A decimal string ('123.4') is in degrees.
      - A sexagesimal string ('12:34:56.7' or '12 34 56.7')
        has `sexagesimal_unit`, which defaults to degrees.

    Parameters
    ----------
    s : :class:`~astropy.coordinates.Angle` or equivalent, string, float
        Anything accepted by `Angle` and also unitless strings and numbers
    sexagesimal_unit : :class:`~astropy.units.UnitBase` or str, optional
        The unit applied to sexagesimal strings

    Returns
    -------
    angle : :class:`~astropy.coordinates.Angle`
        Astropy `Angle`
    """
    try:
        return Angle(s)
    except u.UnitsError:
        # Bytes is a sequence of ints that will inadvertently
        # end up as radians, so crash instead.
        if isinstance(s, bytes):
            raise TypeError(
                f"Raw bytes {s} not supported: " "first decode to string (or add unit)"
            ) from None
        # We now have a number or a string without a unit
        try:
            # Check if it's just a single number
            float(s)
        except ValueError:
            # If not, we assume it's a sexagesimal string that can be parsed by Astropy
            return Angle(s, unit=sexagesimal_unit)
        except TypeError:
            # Postpone dealing with things like NumPy ndarrays
            pass
        if isinstance(s, str):
            return Angle(s, unit=u.deg)
        else:
            # XXX Maybe deprecate this in future and only deal with strings here
            return Angle(s, unit=u.rad)


def strip_zeros(str_or_array_of_str):
    """Remove trailing zeros and unnecessary decimal points from numerical strings."""
    s = np.char.rstrip(str_or_array_of_str, "0")
    s = np.char.rstrip(s, ".")
    return s if s.ndim else s.item()


def angle_to_string(angle, show_unit=True, **kwargs):
    """Convert an Angle to string(s) while maintaining precision and compatibility.

    This serialises angles to strings with high precision (1 micron @ 13000 km)
    while maintaining some compatibility with older katpoint angle strings.
    The main difference is that the numerical representation (sexagesimal or
    decimal) has a suffix indicating the unit ('d' for degree or 'h' for hour).
    This allows Astropy Angles to be constructed directly from it without the
    need for :func:`to_angle`. This suffix can be suppressed when generating
    strings for display purposes.

    Extra keyword arguments are passed on to
    :meth:`~astropy.coordinates.Angle.to_string` to control the appearance
    of the string to some extent, but there are restrictions. The only supported
    units are degree, hour and hourangle. The sexagesimal separator is fixed
    to ':' but ignored when a decimal representation is selected instead
    (unlike in Astropy >= 5.0).

    Parameters
    ----------
    angle : :class:`~astropy.coordinates.Angle`
        An `Angle` object (may be multidimensional)
    show_unit : bool, optional
        True if the unit of the angle ('h' or 'd') is appended to the string
    kwargs : dict, optional
        Extra keyword arguments for :meth:`~astropy.coordinates.Angle.to_string`

    Returns
    -------
    s : str or array of str
        String(s) representing `angle`

    Raises
    ------
    ValueError
        If angle / kwargs unit is not supported, or separator is not ':'
    """
    unit = kwargs.setdefault("unit", angle.unit)
    if unit not in (u.deg, u.hour, u.hourangle):
        raise ValueError(
            f"The angle unit should be degree, hour or hourangle, not {unit}"
        )
    sep = kwargs.setdefault("sep", ":")
    decimal = kwargs.get("decimal", False)
    # Ignore sexagesimal separator when using a decimal representation,
    # instead of crashing on Astropy >= 5.0.
    if decimal:
        del kwargs["sep"]
    elif sep != ":":
        raise ValueError(f"The sexagesimal separator should be ':', not '{sep}'")
    precision = kwargs.get("precision")
    if precision is None:
        # Sufficient precision to discern 1 micron at 13000 km
        precision = 12 if decimal else 8
        # Hour angle needs a bit more precision
        if unit != u.deg:
            precision += 1
        kwargs["precision"] = precision
    number = strip_zeros(angle.to_string(**kwargs))
    if not show_unit:
        return number
    suffix = "d" if unit == u.deg else "h"
    s = np.char.add(number, suffix)
    return s if s.ndim else s.item()


# -------------------------------------------------------------------------------------
# --- Geodetic coordinate transformations
# -------------------------------------------------------------------------------------


def lla_to_ecef(lat_rad, lon_rad, alt_m):
    """Convert WGS84 spherical coordinates to ECEF cartesian coordinates.

    This converts a position on the Earth specified in geodetic latitude,
    longitude and altitude to earth-centered, earth-fixed (ECEF) cartesian
    coordinates. This code assumes the WGS84 earth model, described in
    [NIMA2004]_.

    Parameters
    ----------
    lat_rad : float or array
        Latitude (customary geodetic, not geocentric), in radians
    lon_rad : float or array
        Longitude, in radians
    alt_m : float or array
        Altitude, in metres above WGS84 ellipsoid

    Returns
    -------
    x_m, y_m, z_m : float or array
        X, Y, Z coordinates, in metres

    References
    ----------
    .. [NIMA2004] National Imagery and Mapping Agency, "Department of Defense
       World Geodetic System 1984," NIMA TR8350.2, Page 4-4, last updated
       June, 2004.
    """
    # WGS84 Defining Parameters
    a = 6378137.0  # semi-major axis of Earth in m
    f = 1.0 / 298.257223563  # flattening of Earth

    # WGS84 derived geometric constants
    e2 = 2 * f - f**2  # first eccentricity squared

    # intermediate calculation
    # (normal, or prime vertical radius of curvature)
    R = a / np.sqrt(1.0 - e2 * np.sin(lat_rad) ** 2)

    x_m = (R + alt_m) * np.cos(lat_rad) * np.cos(lon_rad)
    y_m = (R + alt_m) * np.cos(lat_rad) * np.sin(lon_rad)
    z_m = ((1.0 - e2) * R + alt_m) * np.sin(lat_rad)

    return x_m, y_m, z_m


def ecef_to_lla(x_m, y_m, z_m):
    """Convert ECEF cartesian coordinates to WGS84 spherical coordinates.

    This converts an earth-centered, earth-fixed (ECEF) cartesian position to a
    position on the Earth specified in geodetic latitude, longitude and altitude.
    This code assumes the WGS84 earth model.

    Parameters
    ----------
    x_m, y_m, z_m : float or array
        X, Y, Z coordinates, in metres

    Returns
    -------
    lat_rad : float or array
        Latitude (customary geodetic, not geocentric), in radians
    lon_rad : float or array
        Longitude, in radians
    alt_m : float or array
        Altitude, in metres above WGS84 ellipsoid

    Notes
    -----
    Based on the most accurate algorithm according to Zhu [zhu]_, which is
    summarised by Kaplan [kaplan]_ and described in the Wikipedia entry [geo]_.

    .. [zhu] J. Zhu, "Conversion of Earth-centered Earth-fixed coordinates to
       geodetic coordinates," Aerospace and Electronic Systems, IEEE Transactions
       on, vol. 30, pp. 957-961, 1994.
    .. [kaplan] Kaplan, "Understanding GPS: principles and applications," 1 ed.,
       Norwood, MA 02062, USA: Artech House, Inc, 1996.
    .. [geo] Wikipedia entry, "Geodetic system", 2009.
    """
    # WGS84 Defining Parameters
    a = 6378137.0  # semi-major axis of Earth in m
    f = 1.0 / 298.257223563  # flattening of Earth

    # WGS84 derived geometric constants
    b = a * (1.0 - f)  # semi-minor axis in m
    e2 = 2 * f - f**2  # first eccentricity squared
    ep2 = f * (2.0 - f) / (1.0 - f) ** 2  # second eccentricity squared

    # Define squared terms for convenience
    a2, b2 = a**2, b**2
    x2, y2, z2 = x_m**2, y_m**2, z_m**2

    r = np.sqrt(x2 + y2)
    E2 = a2 - b2
    F = 54.0 * b2 * z2
    G = r**2 + (1 - e2) * z2 - e2 * E2
    C = (e2**2 * F * r**2) / (G**3)
    S = (1.0 + C + np.sqrt(C**2 + 2 * C)) ** (1.0 / 3.0)
    P = F / (3.0 * (S + 1.0 / S + 1.0) ** 2 * G**2)
    Q = np.sqrt(1.0 + 2.0 * e2**2 * P)
    r0 = -P * e2 * r / (1.0 + Q) + np.sqrt(
        0.5 * a2 * (1.0 + 1.0 / Q)
        - P * (1 - e2) * z2 / (Q * (1.0 + Q))
        - 0.5 * P * r**2
    )
    U = np.sqrt((r - e2 * r0) ** 2 + z2)
    V = np.sqrt((r - e2 * r0) ** 2 + (1.0 - e2) * z2)
    z0 = (b2 * z_m) / (a * V)
    alt_m = U * (1.0 - b2 / (a * V))
    lat_rad = np.arctan2(z_m + ep2 * z0, r)
    lon_rad = np.arctan2(y_m, x_m)

    return lat_rad, lon_rad, alt_m


def ecef_to_lla2(x_m, y_m, z_m):
    """Convert ECEF cartesian coordinates to WGS84 spherical coordinates.

    This converts an earth-centered, earth-fixed (ECEF) cartesian position to a
    position on the Earth specified in geodetic latitude, longitude and altitude.
    This code assumes the WGS84 earth model.

    Parameters
    ----------
    x_m, y_m, z_m : float or array
        X, Y, Z coordinates, in metres

    Returns
    -------
    lat_rad : float or array
        Latitude (customary geodetic, not geocentric), in radians
    lon_rad : float or array
        Longitude, in radians
    alt_m : float or array
        Altitude, in metres above WGS84 ellipsoid

    Notes
    -----
    This is a copy of the algorithm in the CONRAD codebase (from conradmisclib).
    It's nearly identical to :func:`ecef_to_lla`, but returns lon/lat in
    different ranges.
    """
    # WGS84 ellipsoid constants
    a = 6378137.0  # semi-major axis of Earth in m
    e = 8.1819190842622e-2  # eccentricity of Earth

    b = np.sqrt(a**2 * (1.0 - e**2))
    ep = np.sqrt((a**2 - b**2) / b**2)
    p = np.sqrt(x_m**2 + y_m**2)
    th = np.arctan2(a * z_m, b * p)
    lon_rad = np.arctan2(y_m, x_m)
    lat_rad = np.arctan2(
        (z_m + ep**2 * b * np.sin(th) ** 3), (p - e**2 * a * np.cos(th) ** 3)
    )
    N = a / np.sqrt(1.0 - e**2 * np.sin(lat_rad) ** 2)
    alt_m = p / np.cos(lat_rad) - N

    # Return lon_rad in range [0, 2*pi)
    lon_rad = np.mod(lon_rad, 2.0 * np.pi)

    # Correct for numerical instability in altitude near exact poles
    # (after this correction, error is about 2 millimeters, which is about
    # the same as the numerical precision of the overall function)
    if np.isscalar(alt_m):
        if (abs(x_m) < 1.0) and (abs(y_m) < 1.0):
            alt_m = abs(z_m) - b
    else:
        near_poles = (np.abs(x_m) < 1.0) & (np.abs(y_m) < 1.0)
        alt_m[near_poles] = np.abs(z_m[near_poles]) - b

    return lat_rad, lon_rad, alt_m


def enu_to_ecef(ref_lat_rad, ref_lon_rad, ref_alt_m, e_m, n_m, u_m):
    """Convert ENU coordinates relative to reference location to ECEF coordinates.

    This converts local east-north-up (ENU) coordinates relative to a given
    reference position to earth-centered, earth-fixed (ECEF) cartesian
    coordinates. The reference position is specified by its geodetic latitude,
    longitude and altitude.

    Parameters
    ----------
    ref_lat_rad, ref_lon_rad : float or array
        Geodetic latitude and longitude of reference position, in radians
    ref_alt_m : float or array
        Geodetic altitude of reference position, in metres above WGS84 ellipsoid
    e_m, n_m, u_m : float or array
        East, North, Up coordinates, in metres

    Returns
    -------
    x_m, y_m, z_m : float or array
        X, Y, Z coordinates, in metres
    """
    # ECEF coordinates of reference point
    ref_x_m, ref_y_m, ref_z_m = lla_to_ecef(ref_lat_rad, ref_lon_rad, ref_alt_m)
    sin_lat, cos_lat = np.sin(ref_lat_rad), np.cos(ref_lat_rad)
    sin_lon, cos_lon = np.sin(ref_lon_rad), np.cos(ref_lon_rad)

    x_m = ref_x_m - sin_lon * e_m - sin_lat * cos_lon * n_m + cos_lat * cos_lon * u_m
    y_m = ref_y_m + cos_lon * e_m - sin_lat * sin_lon * n_m + cos_lat * sin_lon * u_m
    z_m = ref_z_m + cos_lat * n_m + sin_lat * u_m

    return x_m, y_m, z_m


def ecef_to_enu(ref_lat_rad, ref_lon_rad, ref_alt_m, x_m, y_m, z_m):
    """Convert ECEF coordinates to ENU coordinates relative to reference location.

    This converts earth-centered, earth-fixed (ECEF) cartesian coordinates to
    local east-north-up (ENU) coordinates relative to a given reference position.
    The reference position is specified by its geodetic latitude, longitude and
    altitude.

    Parameters
    ----------
    ref_lat_rad, ref_lon_rad : float or array
        Geodetic latitude and longitude of reference position, in radians
    ref_alt_m : float or array
        Geodetic altitude of reference position, in metres above WGS84 ellipsoid
    x_m, y_m, z_m : float or array
        X, Y, Z coordinates, in metres

    Returns
    -------
    e_m, n_m, u_m : float or array
        East, North, Up coordinates, in metres
    """
    # ECEF coordinates of reference point
    ref_x_m, ref_y_m, ref_z_m = lla_to_ecef(ref_lat_rad, ref_lon_rad, ref_alt_m)
    delta_x_m, delta_y_m, delta_z_m = x_m - ref_x_m, y_m - ref_y_m, z_m - ref_z_m
    sin_lat, cos_lat = np.sin(ref_lat_rad), np.cos(ref_lat_rad)
    sin_lon, cos_lon = np.sin(ref_lon_rad), np.cos(ref_lon_rad)

    e_m = -sin_lon * delta_x_m + cos_lon * delta_y_m
    n_m = (
        -sin_lat * cos_lon * delta_x_m
        - sin_lat * sin_lon * delta_y_m
        + cos_lat * delta_z_m
    )
    u_m = (
        cos_lat * cos_lon * delta_x_m
        + cos_lat * sin_lon * delta_y_m
        + sin_lat * delta_z_m
    )

    return e_m, n_m, u_m


# -------------------------------------------------------------------------------------
# --- Spherical coordinate transformations
# -------------------------------------------------------------------------------------


def azel_to_enu(az_rad, el_rad):
    """Convert (az, el) spherical coordinates to unit vector in ENU coordinates.

    This converts horizontal spherical coordinates (azimuth and elevation angle)
    to a unit vector in the corresponding local east-north-up (ENU) coordinate
    system.

    Parameters
    ----------
    az_rad, el_rad : float or array
        Azimuth and elevation angle, in radians

    Returns
    -------
    e, n, u : float or array
        East, North, Up coordinates of unit vector
    """
    sin_az, cos_az = np.sin(az_rad), np.cos(az_rad)
    sin_el, cos_el = np.sin(el_rad), np.cos(el_rad)
    return sin_az * cos_el, cos_az * cos_el, sin_el


def enu_to_azel(east, north, up):
    """Convert vector in ENU coordinates to (az, el) spherical coordinates.

    This converts a vector in the local east-north-up (ENU) coordinate system to
    the corresponding horizontal spherical coordinates (azimuth and elevation
    angle). The ENU coordinates can be in any unit, as the vector length will be
    normalised in the conversion process.

    Parameters
    ----------
    east, north, up : float or array
        East, North, Up coordinates (any unit)

    Returns
    -------
    az_rad, el_rad : float or array
        Azimuth and elevation angle, in radians
    """
    return np.arctan2(east, north), np.arctan2(up, np.hypot(east, north))


def hadec_to_enu(ha_rad, dec_rad, lat_rad):
    """Convert (ha, dec) spherical coordinates to unit vector in ENU coordinates.

    This converts equatorial spherical coordinates (hour angle and declination)
    to a unit vector in the corresponding local east-north-up (ENU) coordinate
    system. The geodetic latitude of the observer is also required.

    Parameters
    ----------
    ha_rad, dec_rad, lat_rad : float or array
        Hour angle, declination and geodetic latitude, in radians

    Returns
    -------
    e, n, u : float or array
        East, North, Up coordinates of unit vector
    """
    sin_ha, cos_ha = np.sin(ha_rad), np.cos(ha_rad)
    sin_dec, cos_dec = np.sin(dec_rad), np.cos(dec_rad)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    return (
        -cos_dec * sin_ha,
        cos_lat * sin_dec - sin_lat * cos_dec * cos_ha,
        sin_lat * sin_dec + cos_lat * cos_dec * cos_ha,
    )


def enu_to_xyz(east, north, up, lat_rad):
    """Convert ENU to XYZ coordinates.

    This converts a vector in the local east-north-up (ENU) coordinate system to
    the XYZ coordinate system used in radio astronomy (see e.g. [TMS]_). The X
    axis is the intersection of the equatorial plane and the meridian plane
    through the reference point of the ENU system (and therefore is similar to
    'up'). The Y axis also lies in the equatorial plane to the east of X, and
    coincides with 'east'. The Z axis points toward the north pole, and therefore
    is similar to 'north'. The XYZ system is therefore a local version of the
    Earth-centred Earth-fixed (ECEF) system.

    Parameters
    ----------
    east, north, up : float or array
        East, North, Up coordinates of input vector
    lat_rad : float or array
        Geodetic latitude of ENU / XYZ reference point, in radians

    Returns
    -------
    x, y, z : float or array
        X, Y, Z coordinates of output vector

    References
    ----------
    .. [TMS] Thompson, Moran, Swenson, "Interferometry and Synthesis in Radio
       Astronomy," 2nd ed., Wiley-VCH, 2004, pp. 86-89.
    """
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    return (-sin_lat * north + cos_lat * up, east, cos_lat * north + sin_lat * up)
