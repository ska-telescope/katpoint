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

"""Target object used for pointing and flux density calculation."""

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord, Angle
from astropy.coordinates import ICRS, Galactic, FK4, AltAz, CIRS
from astropy.time import Time

from .timestamp import Timestamp, delta_seconds
from .flux import FluxDensityModel
from .conversion import azel_to_enu
from .projection import sphere_to_plane, sphere_to_ortho, plane_to_sphere
from .body import (Body, FixedBody, SolarSystemBody, EarthSatelliteBody,
                   StationaryBody, NullBody, to_angle)
from .stars import STARS


class NonAsciiError(ValueError):
    """Exception when non-ascii characters are found."""
    pass


class Target:
    """A target which can be pointed at by an antenna.

    This is a wrapper around a PyEphem :class:`ephem.Body` that adds flux
    density, alternate names and descriptive tags. For convenience, a default
    antenna and flux frequency can be set, to simplify the calling of pointing
    and flux density methods. These are not stored as part of the target object,
    however.

    The object can be constructed from its constituent components or from a
    description string. The description string contains up to five
    comma-separated fields, with the format::

        <name list>, <tags>, <longitudinal>, <latitudinal>, <flux model>

    The <name list> contains a pipe-separated list of alternate names for the
    target, with the preferred name either indicated by a prepended asterisk or
    assumed to be the first name in the list. The names may contain spaces, and
    the list may be empty. The <tags> field contains a space-separated list of
    descriptive tags for the target. The first tag is mandatory and indicates
    the body type of the target, which should be one of (*azel*, *radec*, *gal*,
    *tle*, *special*, *star*, *xephem*). The longidutinal and latitudinal fields
    are only relevant to *azel* and *radec* targets, in which case they contain
    the relevant coordinates.

    The <flux model> is a space-separated list of numbers used to represent the
    flux density of the target. The first two numbers specify the frequency
    range for which the flux model is valid (in MHz), and the rest of the numbers
    are model coefficients. The <flux model> may be enclosed in parentheses to
    distinguish it from the other fields. An example string is::

        name1 | *name 2, radec cal, 12:34:56.7, -04:34:34.2, (1000.0 2000.0 1.0)

    For *special* and *star* body types, only the target name is required. The
    *special* body name is assumed to be a PyEphem class name, and is typically
    one of the major solar system objects. Alternatively, it could be "Nothing",
    which indicates a dummy target with no position (useful as a placeholder but
    not much else). The *star* name is looked up in the PyEphem star database,
    which contains a modest list of bright stars.

    For *tle* bodies, the final field in the description string should contain
    the three lines of the TLE. If the name list is empty, the target name is
    taken from the TLE instead. The *xephem* body contains a string in XEphem
    EDB database format as the final field, with commas replaced by tildes. If
    the name list is empty, the target name is taken from the XEphem string
    instead.

    When specifying a description string, the rest of the target parameters are
    ignored, except for the default antenna and flux frequency (which do not
    form part of the description string).

    Parameters
    ----------
    body : :class:`ephem.Body` object or :class:`Target` object or string
        Pre-constructed PyEphem Body object to embed in target object, or
        existing target object or description string
    tags : list of strings, or whitespace-delimited string, optional
        Descriptive tags associated with target, starting with its body type
    aliases : list of strings, optional
        Alternate names of target
    flux_model : :class:`FluxDensity` object, optional
        Object encapsulating spectral flux density model
    antenna : :class:`Antenna` object, optional
        Default antenna to use for position calculations
    flux_freq_MHz : float, optional
        Default frequency at which to evaluate flux density, in MHz

    Arguments
    ---------
    name : string
        Name of target

    Raises
    ------
    ValueError
        If description string has the wrong format
    """

    def __init__(self, body, tags=None, aliases=None, flux_model=None, antenna=None, flux_freq_MHz=None):
        if isinstance(body, Target):
            body = body.description
        # If the first parameter is a description string, extract the relevant target parameters from it
        if isinstance(body, str):
            body, tags, aliases, flux_model = construct_target_params(body)
        self.body = body
        self.name = self.body.name
        self.tags = []
        self.add_tags(tags)
        if aliases is None:
            self.aliases = []
        else:
            self.aliases = aliases
        self.flux_model = flux_model
        self.antenna = antenna
        self.flux_freq_MHz = flux_freq_MHz

    def __str__(self):
        """Verbose human-friendly string representation of target object."""
        descr = str(self.name)
        if self.aliases:
            descr += ' (%s)' % (', '.join(self.aliases),)
        descr += ', tags=%s' % (' '.join(self.tags),)
        if 'radec' in self.tags:
            descr += ', %s %s' % (self.body.coord.ra.to_string(unit=u.hour),
                                  self.body.coord.dec.to_string(unit=u.deg))
        if self.body_type == 'azel':
            descr += ', %s %s' % (self.body.coord.az.to_string(unit=u.deg),
                                  self.body.coord.alt.to_string(unit=u.deg))
        if self.body_type == 'gal':
            gal = self.body.compute(Galactic())
            descr += ', %.4f %.4f' % (gal.l.deg, gal.b.deg)
        if self.flux_model is None:
            descr += ', no flux info'
        else:
            descr += ', flux defined for %g - %g MHz' % (self.flux_model.min_freq_MHz, self.flux_model.max_freq_MHz)
            if self.flux_freq_MHz is not None:
                flux = self.flux_model.flux_density(self.flux_freq_MHz)
                if not np.isnan(flux):
                    descr += ', flux=%.1f Jy @ %g MHz' % (flux, self.flux_freq_MHz)
        return descr

    def __repr__(self):
        """Short human-friendly string representation of target object."""
        sub_type = (' (%s)' % self.tags[1]) if (self.body_type == 'xephem') and (len(self.tags) > 1) else ''
        return "<katpoint.Target '%s' body=%s at 0x%x>" % (self.name, self.body_type + sub_type, id(self))

    def __reduce__(self):
        """Custom pickling routine based on description string."""
        return (self.__class__, (self.description,))

    def __eq__(self, other):
        """Equality comparison operator."""
        return self.description == (other.description if isinstance(other, Target) else other)

    def __ne__(self, other):
        """Inequality comparison operator."""
        return not (self == other)

    def __lt__(self, other):
        """Less-than comparison operator (needed for sorting and np.unique)."""
        return self.description < (other.description if isinstance(other, Target) else other)

    def __hash__(self):
        """Base hash on description string, just like equality operator."""
        return hash(self.description)

    def format_katcp(self):
        """String representation if object is passed as parameter to KATCP command."""
        return self.description

    @property
    def body_type(self):
        """Type of target body, as a string tag."""
        return self.tags[0].lower()

    @property
    def description(self):
        """Complete string representation of target object, sufficient to reconstruct it."""
        names = ' | '.join([self.name] + self.aliases)
        tags = ' '.join(self.tags)
        fluxinfo = self.flux_model.description if self.flux_model is not None else None
        fields = [names, tags]
        if self.body_type == 'azel':
            # Check if it's an unnamed target with a default name
            if names.startswith('Az:'):
                fields = [tags]
            fields += [self.body.coord.az.to_string(unit=u.deg),
                       self.body.coord.alt.to_string(unit=u.deg)]
            if fluxinfo:
                fields += [fluxinfo]

        elif self.body_type == 'radec':
            # Check if it's an unnamed target with a default name
            if names.startswith('Ra:'):
                fields = [tags]
            fields += [self.body.coord.ra.to_string(unit=u.hour),
                       self.body.coord.dec.to_string(unit=u.deg)]
            if fluxinfo:
                fields += [fluxinfo]

        elif self.body_type == 'gal':
            # Check if it's an unnamed target with a default name
            if names.startswith('Galactic l:'):
                fields = [tags]
            gal = self.body.compute(Galactic())
            fields += ['%.4f' % (gal.l.deg,), '%.4f' % (gal.b.deg,)]
            if fluxinfo:
                fields += [fluxinfo]

        elif self.body_type == 'tle':
            fields += self.body.to_tle()
            if fluxinfo:
                fields += [fluxinfo]

        elif self.body_type == 'xephem':
            # Replace commas in xephem string with tildes, to avoid clashing with main string structure
            # Also remove extra spaces added into string by to_edb
            edb_string = '~'.join([edb_field.strip() for edb_field in self.body.to_edb().split(',')])
            # Suppress name if it's the same as in the xephem db string
            edb_name = edb_string[:edb_string.index('~')]
            if edb_name == names:
                fields = [tags]
            fields += [edb_string]

        return ', '.join(fields)

    def add_tags(self, tags):
        """Add tags to target object.

        This adds tags to a target, while checking the sanity of the tags. It
        also prevents duplicate tags without resorting to a tag set, which would
        be problematic since the tag order is meaningful (tags[0] is the body
        type). Since tags should not contain whitespace, any string consisting of
        whitespace-delimited words will be split into separate tags.

        Parameters
        ----------
        tags : string, list of strings, or None
            Tag or list of tags to add (strings will be split on whitespace)

        Returns
        -------
        target : :class:`Target` object
            Updated target object
        """
        if tags is None:
            tags = []
        if isinstance(tags, str):
            tags = [tags]
        for tag_str in tags:
            for tag in tag_str.split():
                if tag not in self.tags:
                    self.tags.append(tag)
        return self

    def _normalise_antenna(self, antenna, required=False):
        """Set default antenna if unspecified and check that antenna is valid.

        If `antenna` is `None`, it is replaced by the default antenna for the
        target (which could also be `None`). Raise a :class:`ValueError` if
        an antenna is required and none is provided.

        Parameters
        ----------
        antenna : :class:`Antenna` or None
            Antenna which points at target
        required : bool, optional
            True if it is an error to have no valid antenna

        Returns
        -------
        antenna : :class:`Antenna` or None
            Antenna which points at target (not None if `required` is True)
        location : :class:`~astropy.coordinates.EarthLocation` or None
            Location of antenna on Earth (not None if `required` is True)

        Raises
        ------
        ValueError
            If an antenna is required and none could be found
        """
        if antenna is None:
            antenna = self.antenna
        if required and antenna is None:
            raise ValueError('Antenna object needed to calculate target position')
        location = antenna.earth_location if antenna is not None else None
        return antenna, location

    def azel(self, timestamp=None, antenna=None):
        """Calculate target (az, el) coordinates as seen from antenna at time(s).

        Parameters
        ----------
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equivalent, optional
            Timestamp(s), defaults to now
        antenna : :class:`Antenna` object, optional
            Antenna which points at target (defaults to default antenna)

        Returns
        -------
        azel : :class:`~astropy.coordinates.AltAz`, same shape as *timestamp*
            Azimuth and elevation in `AltAz` frame

        Raises
        ------
        ValueError
            If no antenna is specified and body type requires it for (az, el)
        """
        time = Timestamp(timestamp).time
        _, location = self._normalise_antenna(antenna)
        altaz = AltAz(obstime=time, location=location)
        return self.body.compute(altaz, obstime=time, location=location)

    def apparent_radec(self, timestamp=None, antenna=None):
        """Calculate target's apparent (ra, dec) coordinates as seen from antenna at time(s).

        This calculates the *apparent topocentric position* of the target for
        the epoch-of-date in equatorial coordinates. Take note that this is
        *not* the "star-atlas" position of the target, but the position as is
        actually seen from the antenna at the given times. The difference is on
        the order of a few arcminutes. These are the coordinates that a telescope
        with an equatorial mount would use to track the target.

        Parameters
        ----------
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equivalent, optional
            Timestamp(s), defaults to now
        antenna : :class:`Antenna` object, optional
            Antenna which points at target (defaults to default antenna)

        Returns
        -------
        radec : :class:`~astropy.coordinates.CIRS`, same shape as *timestamp*
            Right ascension and declination in CIRS frame

        Raises
        ------
        ValueError
            If no antenna is specified and body type requires it for (ra, dec)
        """
        time = Timestamp(timestamp).time
        _, location = self._normalise_antenna(antenna)
        return self.body.compute(CIRS(obstime=time), obstime=time, location=location)

    def astrometric_radec(self, timestamp=None, antenna=None):
        """Calculate target's astrometric (ra, dec) coordinates as seen from antenna at time(s).

        This calculates the ICRS *astrometric barycentric position* of the
        target, in equatorial coordinates. This is its star atlas position for
        the epoch of J2000, as seen from the Solar System barycentre (also
        called "catalog coordinates" in SOFA).

        Parameters
        ----------
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equivalent, optional
            Timestamp(s), defaults to now
        antenna : :class:`Antenna` object, optional
            Antenna which points at target (defaults to default antenna)

        Returns
        -------
        radec : :class:`~astropy.coordinates.ICRS`, same shape as *timestamp*
            Right ascension and declination in ICRS frame

        Raises
        ------
        ValueError
            If no antenna is specified and body type requires it for (ra, dec)
        """
        time = Timestamp(timestamp).time
        _, location = self._normalise_antenna(antenna)
        return self.body.compute(ICRS(), obstime=time, location=location)

    # The default (ra, dec) coordinates are the astrometric ones
    radec = astrometric_radec

    def galactic(self, timestamp=None, antenna=None):
        """Calculate target's galactic (l, b) coordinates as seen from antenna at time(s).

        This calculates the galactic coordinates of the target, based on the
        ICRS *astrometric barycentric* coordinates. This is its position
        relative to the `Galactic` frame for the epoch of J2000.

        Parameters
        ----------
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equivalent, optional
            Timestamp(s), defaults to now
        antenna : :class:`Antenna` object, optional
            Antenna which points at target (defaults to default antenna)

        Returns
        -------
        lb : :class:`~astropy.coordinates.Galactic`, same shape as *timestamp*
            Galactic longitude, *l*, and latitude, *b*, in `Galactic` frame

        Raises
        ------
        ValueError
            If no antenna is specified and body type requires it for (l, b)
        """
        time = Timestamp(timestamp).time
        _, location = self._normalise_antenna(antenna)
        return self.body.compute(Galactic(), obstime=time, location=location)

    def parallactic_angle(self, timestamp=None, antenna=None):
        """Calculate parallactic angle on target as seen from antenna at time(s).

        This calculates the *parallactic angle*, which is the position angle of
        the observer's vertical on the sky, measured from north toward east.
        This is the angle between the great-circle arc connecting the celestial
        North pole to the target position, and the great-circle arc connecting
        the zenith above the antenna to the target, or the angle between the
        *hour circle* and *vertical circle* through the target, at the given
        timestamp(s).

        Parameters
        ----------
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equivalent, optional
            Timestamp(s), defaults to now
        antenna : :class:`Antenna` object, optional
            Antenna which points at target (defaults to default antenna)

        Returns
        -------
        parangle : :class:`~astropy.coordinates.Angle`, same shape as *timestamp*
            Parallactic angle

        Raises
        ------
        ValueError
            If no antenna is specified, and no default antenna was set either

        Notes
        -----
        The formula can be found in the `AIPS++ glossary`_ or in the SLALIB
        source code (file pa.f, function sla_PA) which is part of the now
        defunct `Starlink project`_.

        .. _`AIPS++ Glossary`: http://www.astron.nl/aips++/docs/glossary/p.html
        .. _`Starlink Project`: http://www.starlink.rl.ac.uk
        """
        time = Timestamp(timestamp).time
        antenna, location = self._normalise_antenna(antenna, required=True)
        # Get apparent hour angle and declination
        radec = self.apparent_radec(time, antenna)
        ha = antenna.local_sidereal_time(time) - radec.ra
        y = np.sin(ha)
        x = np.tan(location.lat.rad) * np.cos(radec.dec) - np.sin(radec.dec) * np.cos(ha)
        return Angle(np.arctan2(y, x))

    def geometric_delay(self, antenna2, timestamp=None, antenna=None):
        """Calculate geometric delay between two antennas pointing at target.

        An incoming plane wavefront travelling along the direction from the
        target to the reference antenna *antenna* arrives at this antenna at the
        given timestamp(s), and *delay* seconds later (or earlier, if *delay* is
        negative) at the second antenna, *antenna2*. This delay is known as the
        *geometric delay*, also represented by the symbol :math:`\tau_g`, and is
        associated with the *baseline* vector from the reference antenna to the
        second antenna. Additionally, the rate of change of the delay at the
        given timestamp(s) is estimated from the change in delay during a short
        interval spanning the timestamp(s).

        Parameters
        ----------
        antenna2 : :class:`Antenna` object
            Second antenna of baseline pair (baseline vector points toward it)
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equivalent, optional
            Timestamp(s), defaults to now
        antenna : :class:`Antenna` object, optional
            First (reference) antenna of baseline pair, which also serves as
            pointing reference (defaults to default antenna)

        Returns
        -------
        delay : float, or array of same shape as *timestamp*
            Geometric delay, in seconds
        delay_rate : float, or array of same shape as *timestamp*
            Rate of change of geometric delay, in seconds per second

        Raises
        ------
        ValueError
            If no reference antenna is specified and no default antenna was set

        Notes
        -----
        This is a straightforward dot product between the unit vector pointing
        from the reference antenna to the target, and the baseline vector
        pointing from the reference antenna to the second antenna, all in local
        ENU coordinates relative to the reference antenna.
        """
        time = Timestamp(timestamp).time
        antenna, _ = self._normalise_antenna(antenna, required=True)
        # Obtain baseline vector from reference antenna to second antenna
        baseline_m = antenna.baseline_toward(antenna2)
        # Obtain direction vector(s) from reference antenna to target, and numerically
        # estimate delay rate from difference across 1-second interval spanning timestamp(s)
        times = time[..., np.newaxis] + delta_seconds([-0.5, 0.0, 0.5])
        azel = self.azel(times, antenna)
        targetdirs = np.array(azel_to_enu(azel.az.rad, azel.alt.rad))
        # Dot product of vectors is w coordinate, and
        # delay is time taken by EM wave to traverse this
        delays = -np.einsum('j,j...', baseline_m, targetdirs) / const.c.to_value(u.m / u.s)
        return delays[..., 1], delays[..., 2] - delays[..., 0]

    def uvw_basis(self, timestamp=None, antenna=None):
        """Calculate the coordinate transformation from local ENU coordinates
        to (u,v,w) coordinates while pointing at target.

        For simple cases, use :meth:`uvw` directly. This method is useful for
        computing (u,v,w) coordinates for all antennas in an array more
        efficiently than calling :meth:`uvw` for each antenna in turn.

        Refer to :meth:`uvw` for details about how the (u,v,w) coordinate
        system is defined.

        Parameters
        ----------
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equivalent, optional
            Timestamp(s), defaults to now
        antenna : :class:`Antenna` object, optional
            Reference antenna of baseline pairs, which also serves as
            pointing reference (defaults to default antenna)

        Returns
        -------
        uvw : 2D or 3D array
            Orthogonal basis vectors for the transformation. If `timestamp` is
            scalar, the return value is a matrix to multiply by ENU column
            vectors to produce UVW vectors. If `timestamp` is a vector,
            the first two dimensions correspond to the matrix and the final
            dimension to the timestamp.
        """
        time = Timestamp(timestamp).time
        antenna, _ = self._normalise_antenna(antenna, required=True)
        if not time.isscalar and self.body_type != 'radec':
            # Some calculations depend on ra/dec in a way that won't easily
            # vectorise.
            bases = [self.uvw_basis(t, antenna) for t in time.ravel()]
            return np.stack(bases, axis=-1).reshape(3, 3, *time.shape)

        # Offset the target slightly in declination to approximate the
        # derivative of ENU in the direction of increasing declination. This
        # used to just use the NCP, but the astrometric-to-topocentric
        # conversion doesn't simply rotate the celestial sphere, but also
        # distorts it, and so that introduced errors.
        #
        # To avoid issues close to the poles, we always offset towards the
        # equator. We also can't offset by too little, as ephem uses only
        # single precision and this method suffers from loss of precision.
        # 0.03 was found by experimentation (albeit on a single data set) to
        # to be large enough to avoid the numeric instability.
        if not time.isscalar:
            # Due to the test above, this is a radec target and so timestamp
            # doesn't matter. But we want a scalar.
            radec = self.radec(None, antenna)
        else:
            radec = self.radec(time, antenna)
        offset_sign = -1 if radec.dec > 0 else 1
        offset = construct_radec_target(radec.ra.rad, radec.dec.rad + 0.03 * offset_sign)
        # Get offset az-el vector at current epoch pointed to by reference antenna
        offset_azel = offset.azel(time, antenna)
        # enu vector pointing from reference antenna to offset point
        z = np.array(azel_to_enu(offset_azel.az.rad, offset_azel.alt.rad))
        # Obtain direction vector(s) from reference antenna to target
        azel = self.azel(time, antenna)
        # w axis points toward target
        w = np.array(azel_to_enu(azel.az.rad, azel.alt.rad))
        # u axis is orthogonal to z and w
        u = np.cross(z, w, axis=0) * offset_sign
        u /= np.linalg.norm(u, axis=0)
        # v axis completes the orthonormal basis
        v = np.cross(w, u, axis=0)
        return np.array([u, v, w])

    def uvw(self, antenna2, timestamp=None, antenna=None):
        """Calculate (u,v,w) coordinates of baseline while pointing at target.

        Calculate the (u,v,w) coordinates of the baseline vector from *antenna*
        toward *antenna2*. The w axis points from the first antenna toward the
        target. The v axis is perpendicular to it and lies in the plane passing
        through the w axis and the poles of the earth, on the northern side of w.
        The u axis is perpendicular to the v and w axes, and points to the east
        of w.

        Parameters
        ----------
        antenna2 : :class:`Antenna` object or sequence
            Second antenna of baseline pair (baseline vector points toward it)
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equivalent, optional
            Timestamp(s), defaults to now
        antenna : :class:`Antenna` object, optional
            First (reference) antenna of baseline pair, which also serves as
            pointing reference (defaults to default antenna)

        Returns
        -------
        u, v, w : float or array
            (u, v, w) coordinates of baseline, in metres. If `timestamp` and/or
            `antenna2` is a sequence, returns an array, with axes in that
            order.

        Notes
        -----
        All calculations are done in the local ENU coordinate system centered on
        the first antenna, as opposed to the traditional XYZ coordinate system.
        This avoids having to convert (az, el) angles to (ha, dec) angles and
        uses linear algebra throughout instead.
        """
        time = Timestamp(timestamp).time
        antenna, _ = self._normalise_antenna(antenna, required=True)
        # Obtain basis vectors
        basis = self.uvw_basis(time, antenna)
        # Obtain baseline vector from reference antenna to second antenna
        try:
            baseline_m = np.stack([antenna.baseline_toward(a2) for a2 in antenna2])
        except TypeError:
            baseline_m = antenna.baseline_toward(antenna2)
        # Apply linear coordinate transformation. A single call np.dot won't
        # work for both the scalar and array case, so we explicitly specify the
        # axes to sum over.
        u, v, w = np.tensordot(basis, baseline_m, ([1], [-1]))
        return u, v, w

    def lmn(self, ra, dec, timestamp=None, antenna=None):
        """Calculate (l, m, n) coordinates for another target, while pointing at
        this target.

        Refer to :meth:`uvw` for a description of the coordinate system. This
        function is vectorised, allowing for multiple targets and multiple
        timestamps.

        Parameters
        ----------
        ra : float or array
            Right ascension of the other target, in radians
        dec : float or array
            Declination of the other target, in radians
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equivalent, optional
            Timestamp(s), defaults to now
        antenna : :class:`Antenna` object, optional
            Pointing reference (defaults to default antenna)

        Returns
        -------
        l,m,n : float, or array of same length as `ra`, `dec`, `timestamps`
            (l, m, n) coordinates of target(s).
        """
        ref_radec = self.radec(timestamp, antenna)
        return sphere_to_ortho(ref_radec.ra.rad, ref_radec.dec.rad, ra, dec)

    def flux_density(self, flux_freq_MHz=None):
        """Calculate flux density for given observation frequency (or frequencies).

        This uses the stored flux density model to calculate the flux density at
        a given frequency (or frequencies). See the documentation of
        :class:`FluxDensityModel` for more details of this model. If the flux
        frequency is unspecified, the default value supplied to the target object
        during construction is used. If no flux density model is available or a
        frequency is out of range, a flux value of NaN is returned for that
        frequency.

        This returns only Stokes I. Use :meth:`flux_density_stokes` to get
        polarisation information.

        Parameters
        ----------
        freq_MHz : float or sequence, optional
            Frequency at which to evaluate flux density, in MHz

        Returns
        -------
        flux_density : float, or array of same shape as *freq_MHz*
            Flux density in Jy, or np.nan if frequency is out of range or target
            does not have flux model

        Raises
        ------
        ValueError
            If no frequency is specified, and no default frequency was set either
        """
        if flux_freq_MHz is None:
            flux_freq_MHz = self.flux_freq_MHz
        if flux_freq_MHz is None:
            raise ValueError('Please specify frequency at which to measure flux density')
        if self.flux_model is None:
            # Target has no specified flux density
            flux = np.full(np.shape(flux_freq_MHz), np.nan)
            return flux if flux.ndim else flux.item()
        return self.flux_model.flux_density(flux_freq_MHz)

    def flux_density_stokes(self, flux_freq_MHz=None):
        """Calculate flux density for given observation frequency (or frequencies), full-Stokes.

        See :meth:`flux_density`
        This uses the stored flux density model to calculate the flux density at
        a given frequency (or frequencies). See the documentation of
        :class:`FluxDensityModel` for more details of this model. If the flux
        frequency is unspecified, the default value supplied to the target object
        during construction is used. If no flux density model is available or a
        frequency is out of range, a flux value of NaN is returned for that
        frequency.

        Parameters
        ----------
        freq_MHz : float or sequence, optional
            Frequency at which to evaluate flux density, in MHz

        Returns
        -------
        flux_density : array of float
            Flux density in Jy, or np.nan if frequency is out of range or target
            does not have flux model. The shape matches the input with an extra
            trailing dimension of size 4 containing Stokes I, Q, U, V.

        Raises
        ------
        ValueError
            If no frequency is specified, and no default frequency was set either
        """
        if flux_freq_MHz is None:
            flux_freq_MHz = self.flux_freq_MHz
        if flux_freq_MHz is None:
            raise ValueError('Please specify frequency at which to measure flux density')
        if self.flux_model is None:
            return np.full(np.shape(flux_freq_MHz) + (4,), np.nan)
        return self.flux_model.flux_density_stokes(flux_freq_MHz)

    def separation(self, other_target, timestamp=None, antenna=None):
        """Angular separation between this target and another as viewed from antenna.

        Parameters
        ----------
        other_target : :class:`Target` object
            The other target
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equivalent, optional
            Timestamp(s) when separation is measured (defaults to now)
        antenna : class:`Antenna` object, optional
            Antenna that observes both targets, from where separation is measured
            (defaults to default antenna of this target)

        Returns
        -------
        separation : :class:`~astropy.coordinates.Angle`
            Angular separation between the targets, as viewed from antenna

        Notes
        -----
        This calculates the azimuth and elevation of both targets at the given
        time and finds the angular distance between the two sets of coordinates.
        """
        # Get a common timestamp and antenna for both targets
        time = Timestamp(timestamp).time
        antenna, _ = self._normalise_antenna(antenna)
        this_azel = self.azel(time, antenna)
        other_azel = other_target.azel(time, antenna)
        # XXX Work around Astropy 4.1rc1 limitation (should be fixed in 4.1):
        # SkyCoord.separation triggers a frame equivalency check and this crashes
        # if one azel has an attribute that the other lacks. The workaround blanks
        # out attributes introduced by SolarSystemBody's GCRS coordinates, which
        # should not affect separation calculation.
        # This is fixed by astropy/astropy#10658, slated for the 4.1 milestone.
        this_azel.obsgeoloc = other_azel.obsgeoloc = None
        this_azel.obsgeovel = other_azel.obsgeovel = None
        return this_azel.separation(other_azel)

    def sphere_to_plane(self, az, el, timestamp=None, antenna=None, projection_type='ARC', coord_system='azel'):
        """Project spherical coordinates to plane with target position as reference.

        This is a convenience function that projects spherical coordinates to a
        plane with the target position as the origin of the plane. The function is
        vectorised and can operate on single or multiple timestamps, as well as
        single or multiple coordinate vectors. The spherical coordinates may be
        (az, el) or (ra, dec), and the projection type can also be specified.

        Parameters
        ----------
        az : float or array
            Azimuth or right ascension, in radians
        el : float or array
            Elevation or declination, in radians
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equivalent, optional
            Timestamp(s), defaults to now
        antenna : :class:`Antenna` object, optional
            Antenna pointing at target (defaults to default antenna)
        projection_type : {'ARC', 'SIN', 'TAN', 'STG', 'CAR', 'SSN'}, optional
            Type of spherical projection
        coord_system : {'azel', 'radec'}, optional
            Spherical coordinate system

        Returns
        -------
        x : float or array
            Azimuth-like coordinate(s) on plane, in radians
        y : float or array
            Elevation-like coordinate(s) on plane, in radians
        """
        if coord_system == 'radec':
            # The target (ra, dec) coordinates will serve as reference point on the sphere
            ref_radec = self.radec(timestamp, antenna)
            return sphere_to_plane[projection_type](ref_radec.ra.rad, ref_radec.dec.rad, az, el)
        else:
            # The target (az, el) coordinates will serve as reference point on the sphere
            ref_azel = self.azel(timestamp, antenna)
            return sphere_to_plane[projection_type](ref_azel.az.rad, ref_azel.alt.rad, az, el)

    def plane_to_sphere(self, x, y, timestamp=None, antenna=None, projection_type='ARC', coord_system='azel'):
        """Deproject plane coordinates to sphere with target position as reference.

        This is a convenience function that deprojects plane coordinates to a
        sphere with the target position as the origin of the plane. The function is
        vectorised and can operate on single or multiple timestamps, as well as
        single or multiple coordinate vectors. The spherical coordinates may be
        (az, el) or (ra, dec), and the projection type can also be specified.

        Parameters
        ----------
        x : float or array
            Azimuth-like coordinate(s) on plane, in radians
        y : float or array
            Elevation-like coordinate(s) on plane, in radians
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equivalent, optional
            Timestamp(s), defaults to now
        antenna : :class:`Antenna` object, optional
            Antenna pointing at target (defaults to default antenna)
        projection_type : {'ARC', 'SIN', 'TAN', 'STG', 'CAR', 'SSN'}, optional
            Type of spherical projection
        coord_system : {'azel', 'radec'}, optional
            Spherical coordinate system

        Returns
        -------
        az : float or array
            Azimuth or right ascension, in radians
        el : float or array
            Elevation or declination, in radians
        """
        if coord_system == 'radec':
            # The target (ra, dec) coordinates will serve as reference point on the sphere
            ref_radec = self.radec(timestamp, antenna)
            return plane_to_sphere[projection_type](ref_radec.ra.rad, ref_radec.dec.rad, x, y)
        else:
            # The target (az, el) coordinates will serve as reference point on the sphere
            ref_azel = self.azel(timestamp, antenna)
            return plane_to_sphere[projection_type](ref_azel.az.rad, ref_azel.alt.rad, x, y)

# --------------------------------------------------------------------------------------------------
# --- FUNCTION :  construct_target_params
# --------------------------------------------------------------------------------------------------


def construct_target_params(description):
    """Construct parameters of Target object from description string.

    For more information on the description string format, see the help string
    for :class:`Target`.

    Parameters
    ----------
    description : string
        String containing target name(s), tags, location and flux model

    Returns
    -------
    body : :class:`ephem.Body` object
        PyEphem Body object that will be used for position calculations
    tags : list of strings
        Descriptive tags associated with target, starting with its body type
    aliases : list of strings
        Alternate names of target
    flux_model : :class:`FluxDensity` object
        Object encapsulating spectral flux density model

    Raises
    ------
    ValueError
        If *description* has the wrong format
    """
    try:
        description.encode('ascii')
    except UnicodeError:
        raise NonAsciiError("Target description %r contains non-ASCII characters" % description)
    fields = [s.strip() for s in description.split(',')]
    if len(fields) < 2:
        raise ValueError("Target description '%s' must have at least two fields" % description)
    # Check if first name starts with body type tag, while the next field does not
    # This indicates a missing names field -> add an empty name list in front
    body_types = ['azel', 'radec', 'gal', 'tle', 'special', 'star', 'xephem']
    if np.any([fields[0].startswith(s) for s in body_types]) and \
       not np.any([fields[1].startswith(s) for s in body_types]):
        fields = [''] + fields
    # Extract preferred name from name list (starred or first entry), and make the rest aliases
    names = [s.strip() for s in fields[0].split('|')]
    if len(names) == 0:
        preferred_name, aliases = '', []
    else:
        try:
            ind = [name.startswith('*') for name in names].index(True)
            preferred_name, aliases = names[ind][1:], names[:ind] + names[ind + 1:]
        except ValueError:
            preferred_name, aliases = names[0], names[1:]
    tags = [s.strip() for s in fields[1].split(' ')]
    if len(tags) == 0:
        raise ValueError("Target description '%s' needs at least one tag (body type)" % description)
    body_type = tags[0].lower()
    # Remove empty fields starting from the end (useful when parsing CSV files with fixed number of fields)
    while len(fields[-1]) == 0:
        fields.pop()

    # Create appropriate PyEphem body based on body type
    if body_type == 'azel':
        if len(fields) < 4:
            raise ValueError("Target description '%s' contains *azel* body with no (az, el) coordinates"
                             % description)
        body = StationaryBody(fields[2], fields[3], preferred_name)

    elif body_type == 'radec':
        if len(fields) < 4:
            raise ValueError("Target description '%s' contains *radec* body with no (ra, dec) coordinates"
                             % description)
        ra = to_angle(fields[2], sexagesimal_unit=u.hour)
        dec = to_angle(fields[3])
        if not preferred_name:
            preferred_name = "Ra: %s Dec: %s" % (ra, dec)
        # Extract epoch info from tags
        if ('B1900' in tags) or ('b1900' in tags):
            frame = FK4(equinox=Time(1900.0, format='byear'))
        elif ('B1950' in tags) or ('b1950' in tags):
            frame = FK4(equinox=Time(1950.0, format='byear'))
        else:
            frame = ICRS
        body = FixedBody(preferred_name, SkyCoord(ra=ra, dec=dec, frame=frame))

    elif body_type == 'gal':
        if len(fields) < 4:
            raise ValueError("Target description '%s' contains *gal* body with no (l, b) coordinates"
                             % description)
        l, b = float(fields[2]), float(fields[3])
        if not preferred_name:
            preferred_name = "Galactic l: %.4f b: %.4f" % (l, b)
        body = FixedBody(preferred_name, SkyCoord(l=Angle(l, unit=u.deg),
                                                  b=Angle(b, unit=u.deg), frame=Galactic))

    elif body_type == 'tle':
        if len(fields) < 4:
            raise ValueError(f"Target description '{description}' contains *tle* body "
                             "without the expected two comma-separated lines")
        if not preferred_name:
            preferred_name = 'Unnamed Satellite'
        try:
            body = EarthSatelliteBody.from_tle(preferred_name, fields[2], fields[3])
        except ValueError as err:
            raise ValueError(f"Target description '{description}' "
                             f"contains malformed *tle* body: {err}") from err

    elif body_type == 'special':
        try:
            if preferred_name.capitalize() != 'Nothing':
                body = SolarSystemBody(preferred_name)
            else:
                body = NullBody()
        except ValueError as err:
            raise ValueError("Target description '%s' contains unknown *special* body '%s'"
                             % (description, preferred_name)) from err

    elif body_type == 'star':
        star_name = ' '.join([w.capitalize() for w in preferred_name.split()])
        try:
            body = STARS[star_name]
        except KeyError:
            raise ValueError(f"Target description '{description}' "
                             f"contains unknown *star* '{star_name}'") from None

    elif body_type == 'xephem':
        edb_string = fields[-1].replace('~', ',')
        edb_name_field = edb_string.partition(',')[0]
        edb_names = [name.strip() for name in edb_name_field.split('|')]
        if preferred_name:
            edb_string = edb_string.replace(edb_name_field, preferred_name)
        else:
            preferred_name = edb_names[0]
        if preferred_name != edb_names[0]:
            aliases.append(edb_names[0])
        for extra_name in edb_names[1:]:
            if not (extra_name in aliases) and not (extra_name == preferred_name):
                aliases.append(extra_name)
        try:
            body = Body.from_edb(edb_string)
        except ValueError as err:
            raise ValueError(f"Target description '{description}' "
                             f"contains malformed *xephem* body: {err}") from err
        # Add xephem body type as an extra tag, right after the main 'xephem' tag
        edb_type = edb_string[edb_string.find(',') + 1]
        if edb_type == 'f':
            tags.insert(1, 'radec')
        elif edb_type in ['e', 'h', 'p']:
            tags.insert(1, 'solarsys')
        elif edb_type == 'E':
            tags.insert(1, 'tle')
        elif edb_type == 'P':
            tags.insert(1, 'special')

    else:
        raise ValueError("Target description '%s' contains unknown body type '%s'" % (description, body_type))

    # Extract flux model if it is available
    flux_model = FluxDensityModel(fields[4]) if (len(fields) > 4) and (len(fields[4].strip(' ()')) > 0) else None

    return body, tags, aliases, flux_model

# --------------------------------------------------------------------------------------------------
# --- FUNCTION :  construct_azel_target
# --------------------------------------------------------------------------------------------------


def construct_azel_target(az, el):
    """Convenience function to create unnamed stationary target (*azel* body type).

    The input parameters will also accept :class:`ephem.Angle` objects, as these
    are floats in radians internally.

    Parameters
    ----------
    az : string or float
        Azimuth, either in 'D:M:S' string format, or as a float in radians
    el : string or float
        Elevation, either in 'D:M:S' string format, or as a float in radians

    Returns
    -------
    target : :class:`Target` object
        Constructed target object
    """
    return Target(StationaryBody(az, el), 'azel')

# --------------------------------------------------------------------------------------------------
# --- FUNCTION :  construct_radec_target
# --------------------------------------------------------------------------------------------------


def construct_radec_target(ra, dec):
    """Convenience function to create unnamed fixed target (*radec* body type).

    The input parameters will also accept :class:`ephem.Angle` objects, as these
    are floats in radians internally. The epoch is assumed to be J2000.

    Parameters
    ----------
    ra : string or float
        Right ascension, either in 'H:M:S' or decimal degree string format, or
        as a float in radians
    dec : string or float
        Declination, either in 'D:M:S' or decimal degree string format, or as
        a float in radians

    Returns
    -------
    target : :class:`Target` object
        Constructed target object
    """
    ra = to_angle(ra, sexagesimal_unit=u.hour)
    dec = to_angle(dec)
    name = "Ra: %s Dec: %s" % (ra, dec)
    body = FixedBody(name, SkyCoord(ra=ra, dec=dec, frame=ICRS))
    return Target(body, 'radec')
