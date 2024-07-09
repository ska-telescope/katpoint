################################################################################
# Copyright (c) 2009-2021,2023, National Research Foundation (SARAO)
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

import warnings
from types import SimpleNamespace

import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.coordinates import (
    CIRS,
    FK4,
    ICRS,
    AltAz,
    Angle,
    CartesianRepresentation,
    Galactic,
    SkyCoord,
)
from astropy.time import Time

from .antenna import Antenna
from .body import (
    Body,
    EarthSatelliteBody,
    FixedBody,
    GalacticBody,
    NullBody,
    SolarSystemBody,
    StationaryBody,
)
from .conversion import angle_to_string, azel_to_enu, to_angle
from .flux import FluxDensityModel
from .projection import plane_to_sphere, sphere_to_ortho, sphere_to_plane
from .timestamp import Timestamp, delta_seconds

# Singleton that identifies default target parameters
_DEFAULT = object()


class NonAsciiError(ValueError):
    """Exception when non-ascii characters are found."""


class Target:
    """A target which can be pointed at by an antenna.

    This is a wrapper around a :class:`Body` that adds alternate names,
    descriptive tags and a flux density model. For convenience, a default
    antenna and flux frequency can be set, to simplify the calling of
    pointing and flux density methods. It is also possible to construct a
    Target directly from an :class:`~astropy.coordinates.SkyCoord`.

    The object can be constructed from its constituent components or from a
    description string. The description string contains up to five
    comma-separated fields, with the format::

        [<name list>,] <tags>, [<location 1>, [<location 2>, [<flux model>]]]

    The <name list> contains a pipe-separated list of alternate names for the
    target, with the preferred name either indicated by a prepended asterisk or
    assumed to be the first name in the list. The names may contain spaces.
    The list may also be empty, or the entire field may be missing, to indicate
    an unnamed target. In this case a name will be inferred from the body.

    The <tags> field contains a space-separated list of descriptive tags for
    the target. The first tag is mandatory and indicates the body type of the
    target, which should be one of (*azel*, *radec*, *gal*, *special*, *tle*,
    *xephem*).

    For *azel*, *radec* and *gal* targets, the two location fields contain the
    relevant longitude and latitude coordinates, respectively. The following
    angle string formats are supported::

      - Decimal, always in degrees (e.g. '12.5')
      - Sexagesimal, in hours for right ascension and degrees for the rest,
        with a colon or space separator (e.g. '12:30:00' or '12 30')
      - Decimal or sexagesimal with explicit unit suffix 'd' or 'h',
        e.g. '12.5h' (hours, not degrees!) or '12:30d'

    The *special* body type has no location fields. The *special* target name
    is typically one of the major solar system objects supported by Astropy's
    ephemeris routines. Alternatively, it could be "Nothing", which indicates a
    dummy target with no position (useful as a placeholder but not much else).

    For *tle* bodies, the two location fields contain the two lines of the
    element set. If the name list is empty, the target name is derived from the
    TLE instead.

    The *xephem* body contains a string in XEphem EDB database format as the
    first location field, with commas replaced by tildes. If the name list is
    empty, the target name is taken from the XEphem string instead. Only fixed
    and Earth satellite objects are supported.

    The <flux model> is a space-separated list of numbers used to represent the
    flux density of the target. The first two numbers specify the frequency
    range for which the flux model is valid (in MHz), and the rest of the numbers
    are model coefficients. The <flux model> may be enclosed in parentheses to
    distinguish it from the other fields. An example string is::

        name1 | *name 2, radec cal, 12:34:56.7, -04:34:34.2, (1000.0 2000.0 1.0)

    The default antenna and flux frequency are not stored in the description string.

    In summary, description strings take the following forms based on body type::

        [<name list>,] azel [<user tags>], <az>, <el> [, <flux model>]
        [<name list>,] radec [<user tags>], <ra>, <dec> [, <flux model>]
        [<name list>,] gal [<user tags>], <l>, <b> [, <flux model>]
        <name list>, special [<user tags>] [, <flux model>]
        [<name list>,] tle [<user tags>], <TLE line 1>, <TLE line 2> [, <flux model>]
        [<name list>,] xephem radec [<user tags>], <EDB string (type 'f')> [, <flux>]
        [<name list>,] xephem tle [<user tags>], <EDB string (type 'E')> [, <flux>]

    Parameters
    ----------
    target : :class:`Body`, :class:`~astropy.coordinates.SkyCoord`, str or
        :class:`Target`
        A Body, a SkyCoord, a full description string or existing Target object.
        The parameters in the description string or existing Target can still
        be overridden by providing additional parameters after `target`.
    name : str, optional
        Preferred name of target (use the default Body name if empty)
    user_tags : sequence of str, or whitespace-delimited str, optional
        Descriptive tags associated with target (not including body type)
    aliases : iterable of str, optional
        Alternate names of target
    flux_model : :class:`FluxDensity`, optional
        Object encapsulating spectral flux density model
    antenna : :class:`~astropy.coordinates.EarthLocation` or :class:`Antenna`, optional
        Default antenna / location to use for position calculations
    flux_frequency : :class:`~astropy.units.Quantity`, optional
        Default frequency at which to evaluate flux density

    Raises
    ------
    ValueError
        If description string has the wrong format
    """

    def __init__(
        self,
        target,
        name=_DEFAULT,
        user_tags=_DEFAULT,
        aliases=_DEFAULT,
        flux_model=_DEFAULT,
        antenna=_DEFAULT,
        flux_frequency=_DEFAULT,
    ):
        default = SimpleNamespace(
            name="",
            user_tags=[],
            aliases=(),
            flux_model=None,
            antenna=None,
            flux_frequency=None,
        )
        if isinstance(target, str):
            # Create a temporary Target object to serve up default parameters instead
            target = Target.from_description(target)
        elif isinstance(target, SkyCoord):
            target = FixedBody(target)
        if isinstance(target, Target):
            default = target
            target = default.body
        if not isinstance(target, Body):
            raise TypeError(
                "Expected a Body, Target, SkyCoord or str input to Target, "
                f"not {target.__class__.__name__}"
            )

        self.body = target
        self._name = default.name if name is _DEFAULT else name
        self.user_tags = []
        user_tags = default.user_tags if user_tags is _DEFAULT else user_tags
        self.add_tags(user_tags)
        self._aliases = default.aliases if aliases is _DEFAULT else tuple(aliases)
        self.flux_model = default.flux_model if flux_model is _DEFAULT else flux_model
        self.antenna = default.antenna if antenna is _DEFAULT else antenna
        self._flux_frequency = None
        self.flux_frequency = (
            default.flux_frequency if flux_frequency is _DEFAULT else flux_frequency
        )

    def __str__(self):
        """Complete string representation of target object."""
        return self.description

    def __repr__(self):
        """Short human-friendly string representation of target object."""
        sub_type = (
            f" ({self.tags[1]})"
            if self.body_type == "xephem" and len(self.tags) > 1
            else ""
        )
        return (
            f"<katpoint.Target '{self.name}' body={self.body_type + sub_type} "
            f"at {id(self):#x}>"
        )

    def __reduce__(self):
        """Pickle object based on description string."""
        return (self.__class__, (self.description,))

    def __eq__(self, other):
        """Equality comparison operator."""
        return self.description == (
            other.description if isinstance(other, Target) else other
        )

    def __lt__(self, other):
        """Less-than comparison operator (needed for sorting and np.unique)."""
        return self.description < (
            other.description if isinstance(other, Target) else other
        )

    def __hash__(self):
        """Compute hash on description string, just like equality operator."""
        return hash(self.description)

    @property
    def tags(self):
        """List of descriptive tags associated with target, body type first."""
        return self.body.tag.split() + self.user_tags

    @property
    def body_type(self):
        """Type of target body, as a string tag."""
        return self.tags[0]

    @property
    def name(self):
        """Preferred name of the target."""
        return self._name if self._name else self.body.default_name

    @property
    def aliases(self):
        """Tuple of alternate names of the target."""
        return self._aliases

    @property
    def names(self):
        """Tuple of all names (both preferred and alternate) of the target."""
        return (self.name,) + self._aliases

    @property
    def flux_frequency(self):
        """Default frequency at which to evaluate flux density."""
        return self._flux_frequency

    @flux_frequency.setter
    @u.quantity_input(equivalencies=u.spectral())
    def flux_frequency(self, frequency: u.Hz = None):
        """Check that frequency has a valid unit or is `None`."""
        self._flux_frequency = frequency

    @property
    def description(self):
        """Complete string representation of target object."""
        names = " | ".join(self.names)
        tags = " ".join(self.tags)
        fluxinfo = self.flux_model.description if self.flux_model is not None else None
        no_name = (
            self.body_type != "special"
            and names == self.body.default_name
            or self.body_type == "xephem"
        )
        fields = [tags] if no_name else [names, tags]

        if self.body_type == "azel":
            fields += [
                angle_to_string(self.body.coord.az, unit=u.deg),
                angle_to_string(self.body.coord.alt, unit=u.deg),
            ]
        elif self.body_type == "radec":
            fields += [
                angle_to_string(self.body.coord.ra, unit=u.hour),
                angle_to_string(self.body.coord.dec, unit=u.deg),
            ]
        elif self.body_type == "gal":
            gal = self.body.coord.galactic
            fields += [
                angle_to_string(gal.l, unit=u.deg, decimal=True),
                angle_to_string(gal.b, unit=u.deg, decimal=True),
            ]
        elif self.body_type == "tle":
            fields += self.body.to_tle()
        elif self.body_type == "xephem":
            # Push the names back into EDB string
            # (or remove them entirely if the Body default)
            edb_names = "" if names == self.body.default_name else names
            # Replace commas in xephem string with tildes
            # to avoid clashes with main structure.
            fields += [self.body.to_edb(edb_names).replace(",", "~")]

        if fluxinfo:
            fields += [fluxinfo]
        return ", ".join(fields)

    @classmethod
    def from_description(cls, description):
        """Construct Target object from description string.

        For more information on the description string format, see the help string
        for :class:`Target`.

        Parameters
        ----------
        description : str
            String containing target name(s), tags, location and flux model

        Returns
        -------
        target : :class:`Target`
            Constructed target object

        Raises
        ------
        ValueError
            If *description* has the wrong format
        """
        prefix = f"Target description '{description}'"
        try:
            description.encode("ascii")
        except UnicodeError as err:
            raise NonAsciiError(f"{prefix} contains non-ASCII characters") from err
        fields = [s.strip() for s in description.split(",")]
        if len(fields) < 2:
            raise ValueError(f"{prefix} must have at least two fields")
        # Check if first name starts with body type tag, while the next field does not
        # This indicates a missing names field -> add an empty name list in front
        body_types = ["azel", "radec", "gal", "special", "tle", "xephem"]

        def tags_in(field):
            return any(field.startswith(s) for s in body_types)

        if tags_in(fields[0]) and not tags_in(fields[1]):
            fields.insert(0, "")
        # Extract preferred name from name list (starred or first entry),
        # and make the rest aliases.
        name_field = fields.pop(0)
        names = [s.strip() for s in name_field.split("|")]
        if len(names) == 0:
            preferred_name, aliases = "", []
        else:
            try:
                ind = [name.startswith("*") for name in names].index(True)
                preferred_name, aliases = names[ind][1:], names[:ind] + names[ind + 1 :]
            except ValueError:
                preferred_name, aliases = names[0], names[1:]
        tag_field = fields.pop(0)
        tags = [s.strip() for s in tag_field.split(" ")]
        if not tags:
            raise ValueError(f"{prefix} needs at least one tag (body type)")
        body_type = tags.pop(0).lower()
        # Remove empty fields starting from the end
        # (useful when parsing CSV files with fixed number of fields)
        while fields and not fields[-1]:
            fields.pop()

        # Create appropriate Body based on body type
        if body_type == "azel":
            if len(fields) < 2:
                raise ValueError(
                    f"{prefix} contains *azel* body with no (az, el) coordinates"
                )
            az = fields.pop(0)
            el = fields.pop(0)
            body = StationaryBody(az, el)

        elif body_type == "radec":
            if len(fields) < 2:
                raise ValueError(
                    f"{prefix} contains *radec* body with no (ra, dec) coordinates"
                )
            ra = to_angle(fields.pop(0), sexagesimal_unit=u.hour)
            dec = to_angle(fields.pop(0))
            # Extract epoch info from tags
            if ("B1900" in tags) or ("b1900" in tags):
                frame = FK4(equinox=Time(1900.0, format="byear"))
            elif ("B1950" in tags) or ("b1950" in tags):
                frame = FK4(equinox=Time(1950.0, format="byear"))
            else:
                frame = ICRS
            body = FixedBody(SkyCoord(ra=ra, dec=dec, frame=frame))

        elif body_type == "gal":
            if len(fields) < 2:
                raise ValueError(
                    f"{prefix} contains *gal* body with no (l, b) coordinates"
                )
            gal_l = to_angle(fields.pop(0))
            gal_b = to_angle(fields.pop(0))
            body = GalacticBody(SkyCoord(l=gal_l, b=gal_b, frame=Galactic))

        elif body_type == "tle":
            if len(fields) < 2:
                raise ValueError(
                    f"{prefix} contains *tle* body without "
                    "the expected two comma-separated lines"
                )
            line1 = fields.pop(0)
            line2 = fields.pop(0)
            try:
                body = EarthSatelliteBody.from_tle(line1, line2)
            except ValueError as err:
                raise ValueError(
                    f"{prefix} contains malformed *tle* body: {err}"
                ) from err

        elif body_type == "special":
            try:
                if preferred_name.capitalize() != "Nothing":
                    body = SolarSystemBody(preferred_name)
                else:
                    body = NullBody()
            except ValueError as err:
                raise ValueError(
                    f"{prefix} contains unknown " f"*special* body '{preferred_name}'"
                ) from err

        elif body_type == "xephem":
            if len(fields) < 1:
                raise ValueError(
                    f"Target description '{description}' contains *xephem* body "
                    "without EDB string"
                )
            edb_string = fields.pop(0).replace("~", ",")
            edb_name_field, comma, edb_coord_fields = edb_string.partition(",")
            edb_names = [name.strip() for name in edb_name_field.split("|")]
            if not preferred_name:
                preferred_name = edb_names[0]
            for edb_name in edb_names:
                if edb_name and edb_name != preferred_name and edb_name not in aliases:
                    aliases.append(edb_name)
            try:
                body = Body.from_edb(comma + edb_coord_fields)
            except ValueError as err:
                raise ValueError(
                    f"{prefix} contains malformed *xephem* body: {err}"
                ) from err

        else:
            raise ValueError(f"{prefix} contains unknown body type '{body_type}'")

        # Extract flux model if it is available
        if fields and fields[0].strip(" ()"):
            flux_model = FluxDensityModel.from_description(fields[0])
        else:
            flux_model = None
        return cls(body, preferred_name, tags, aliases, flux_model)

    @classmethod
    def from_azel(cls, az, el):
        """Create unnamed stationary target (*azel* body type).

        Parameters
        ----------
        az, el : :class:`~astropy.coordinates.Angle` or equivalent, string or float
        Azimuth and elevation, as anything accepted by `Angle`, a sexagesimal or
        decimal string in degrees, or as a float in radians

        Returns
        -------
        target : :class:`Target`
            Constructed target object
        """
        return cls(StationaryBody(az, el))

    @classmethod
    def from_radec(cls, ra, dec):
        """Create unnamed fixed target (*radec* body type, ICRS frame).

        Parameters
        ----------
        ra : :class:`~astropy.coordinates.Angle` or equivalent, string or float
            Right ascension, as anything accepted by `Angle`, a sexagesimal
            string in hours, a decimal string in degrees, or as a float in radians
        dec : :class:`~astropy.coordinates.Angle` or equivalent, string or float
            Declination, as anything accepted by `Angle`, a sexagesimal or
            decimal string in degrees, or as a float in radians

        Returns
        -------
        target : :class:`Target`
            Constructed target object
        """
        ra = to_angle(ra, sexagesimal_unit=u.hour)
        dec = to_angle(dec)
        return cls(FixedBody(SkyCoord(ra=ra, dec=dec, frame=ICRS)))

    def add_tags(self, tags):
        """Add tags to target object.

        This adds tags to a target, while checking the sanity of the tags. It
        also prevents duplicate tags without resorting to a tag set, which would
        be problematic since the tag order is meaningful (tags[0] is the body
        type). Since tags should not contain whitespace, any string consisting of
        whitespace-delimited words will be split into separate tags.

        Parameters
        ----------
        tags : str, list of str, or None
            Tag or list of tags to add (strings will be split on whitespace)

        Returns
        -------
        target : :class:`Target`
            Updated target object
        """
        if tags is None:
            tags = []
        if isinstance(tags, str):
            tags = [tags]
        for tag_str in tags:
            for tag in tag_str.split():
                if tag not in self.tags:
                    self.user_tags.append(tag)
        return self

    def _astropy_funnel(self, timestamp, antenna):
        """Turn time and location objects into their Astropy equivalents.

        Parameters
        ----------
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equivalent
            Timestamp(s) in katpoint or Astropy format
        antenna : :class:`~astropy.coordinates.EarthLocation`, :class:`Antenna` or None
            Antenna location(s) in katpoint or Astropy format (None => default antenna)

        Returns
        -------
        time : :class:`~astropy.time.Time`
            Timestamp(s) in Astropy format
        location : :class:`~astropy.coordinates.EarthLocation` or None
            Antenna location(s) in Astropy format
        """
        time = Timestamp(timestamp).time
        if antenna is None:
            antenna = self.antenna
        location = antenna.location if isinstance(antenna, Antenna) else antenna
        return time, location

    def _valid_antenna(self, antenna):
        """Set default antenna if unspecified and check that antenna is valid.

        If `antenna` is `None`, it is replaced by the default antenna for the
        target (which could also be `None`, raising a :class:`ValueError`).

        Parameters
        ----------
        antenna : :class:`~astropy.coordinates.EarthLocation`, :class:`Antenna` or None
            Antenna which points at target (or equivalent Astropy location)

        Returns
        -------
        antenna : :class:`Antenna`
            A valid katpoint Antenna

        Raises
        ------
        ValueError
            If both `antenna` and default antenna are `None`
        """
        if antenna is None:
            antenna = self.antenna
        if antenna is None:
            raise ValueError("Antenna object needed to calculate target position")
        return Antenna(antenna)

    def azel(self, timestamp=None, antenna=None):
        """Calculate target (az, el) coordinates as seen from antenna at time(s).

        Parameters
        ----------
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equiv, optional
            Timestamp(s), defaults to now
        antenna : :class:`~astropy.coordinates.EarthLocation` or
            :class:`Antenna`, optional
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
        time, location = self._astropy_funnel(timestamp, antenna)
        altaz = AltAz(obstime=time, location=location)
        return self.body.compute(altaz, time, location)

    def apparent_radec(self, timestamp=None, antenna=None):
        """Calculate target's apparent (ra, dec) as seen from antenna at time(s).

        This calculates the *apparent topocentric position* of the target for
        the epoch-of-date in equatorial coordinates. Take note that this is
        *not* the "star-atlas" position of the target, but the position as is
        actually seen from the antenna at the given times. The difference is on
        the order of a few arcminutes. These are the coordinates that a telescope
        with an equatorial mount would use to track the target.

        Parameters
        ----------
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equiv, optional
            Timestamp(s), defaults to now
        antenna : :class:`~astropy.coordinates.EarthLocation` or
            :class:`Antenna`, optional
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
        time, location = self._astropy_funnel(timestamp, antenna)
        # XXX This is a bit of mess... Consider going to TETE
        # for the traditional geocentric apparent place or remove entirely
        return self.body.compute(
            CIRS(obstime=time), time, location, to_celestial_sphere=True
        )

    def astrometric_radec(self, timestamp=None, antenna=None):
        """Calculate target's astrometric (ra, dec) as seen from antenna at time(s).

        This calculates the ICRS *astrometric topocentric position* of the
        target, in equatorial coordinates. This is its star atlas position for
        the epoch of J2000, as seen from the antenna (also called "catalog
        coordinates" in SOFA).

        Parameters
        ----------
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equiv, optional
            Timestamp(s), defaults to now
        antenna : :class:`~astropy.coordinates.EarthLocation` or
            :class:`Antenna`, optional
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
        time, location = self._astropy_funnel(timestamp, antenna)
        return self.body.compute(ICRS(), time, location, to_celestial_sphere=True)

    # The default (ra, dec) coordinates are the astrometric ones
    radec = astrometric_radec

    def galactic(self, timestamp=None, antenna=None):
        """Calculate target's galactic (l, b) as seen from antenna at time(s).

        This calculates the galactic coordinates of the target, based on the
        ICRS *astrometric topocentric* coordinates. This is its position
        relative to the `Galactic` frame for the epoch of J2000 as seen from
        the antenna.

        Parameters
        ----------
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equiv, optional
            Timestamp(s), defaults to now
        antenna : :class:`~astropy.coordinates.EarthLocation` or
            :class:`Antenna`, optional
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
        time, location = self._astropy_funnel(timestamp, antenna)
        return self.body.compute(Galactic(), time, location, to_celestial_sphere=True)

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
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equiv, optional
            Timestamp(s), defaults to now
        antenna : :class:`~astropy.coordinates.EarthLocation` or
            :class:`Antenna`, optional
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
        # XXX Right ascension and local time should use the same framework:
        # either CIRS RA and earth rotation angle, or
        # TETE RA and local sidereal time
        time, location = self._astropy_funnel(timestamp, antenna)
        antenna = self._valid_antenna(antenna)
        # Get apparent hour angle and declination
        radec = self.apparent_radec(time, location)
        ha = antenna.local_sidereal_time(time) - radec.ra
        y = np.sin(ha)
        x = np.tan(location.lat.rad) * np.cos(radec.dec) - np.sin(radec.dec) * np.cos(
            ha
        )
        return Angle(np.arctan2(y, x))

    def geometric_delay(self, antenna2, timestamp=None, antenna=None):
        r"""Calculate geometric delay between two antennas pointing at target.

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
        antenna2 : :class:`~astropy.coordinates.EarthLocation` or :class:`Antenna`
            Second antenna of baseline pair (baseline vector points toward it)
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equiv, optional
            Timestamp(s), defaults to now
        antenna : :class:`~astropy.coordinates.EarthLocation` or
            :class:`Antenna`, optional
            First (reference) antenna of baseline pair, which also serves as
            pointing reference (defaults to default antenna)

        Returns
        -------
        delay : :class:`~astropy.units.Quantity` of same shape as *timestamp*
            Geometric delay
        delay_rate : :class:`~astropy.units.Quantity` of same shape as *timestamp*
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
        time, location = self._astropy_funnel(timestamp, antenna)
        antenna = self._valid_antenna(antenna)
        # Obtain baseline vector from reference antenna to second antenna
        baseline = antenna.baseline_toward(antenna2)
        # Obtain direction vector(s) from reference antenna to target,
        # and numerically estimate delay rate from difference across
        # 1-second interval spanning timestamp(s).
        offset = delta_seconds([-0.5, 0.0, 0.5])
        times = time[..., np.newaxis] + offset
        azel = self.azel(times, location)
        targetdirs = np.array(azel_to_enu(azel.az.rad, azel.alt.rad))
        # Dot product of vectors is w coordinate, and
        # delay is time taken by EM wave to traverse this
        delays = -np.einsum("j,j...", baseline.xyz, targetdirs) / const.c
        delay_rate = (delays[..., 2] - delays[..., 0]) / (offset[2] - offset[0]).to(u.s)
        return delays[..., 1], delay_rate

    def uvw_basis(self, timestamp=None, antenna=None):
        """Calculate ENU -> (u,v,w) transform while pointing at the target.

        Calculate the coordinate transformation from local ENU coordinates
        to (u,v,w) coordinates while pointing at the target.

        In most cases you should use :meth:`uvw` directly.

        Refer to :meth:`uvw` for details about how the (u,v,w) coordinate
        system is defined.

        Parameters
        ----------
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equiv, optional
            Timestamp(s) of shape T, defaults to now
        antenna : :class:`~astropy.coordinates.EarthLocation` or
            :class:`Antenna`, optional
            Reference antenna of baseline pairs, which also serves as
            pointing reference (defaults to default antenna)

        Returns
        -------
        uvw_basis : array of float, shape (3, 3) + T
            Orthogonal basis vectors for the transformation. If `timestamp` is
            scalar, the return value is a matrix to multiply by ENU column
            vectors to produce UVW vectors. If `timestamp` is an array of
            shape T, the first two dimensions correspond to the matrix and the
            remaining dimension(s) to the timestamp.
        """
        time, location = self._astropy_funnel(timestamp, antenna)
        # Check that antenna is valid to avoid more cryptic
        # error messages in .azel and .radec
        self._valid_antenna(antenna)
        if not time.isscalar and self.body_type != "radec":
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
            radec = self.radec(None, location)
        else:
            radec = self.radec(time, location)
        offset_sign = -1 if radec.dec > 0 else 1
        offset = Target.from_radec(radec.ra, radec.dec + offset_sign * 0.03 * u.rad)
        # Get offset az-el vector at current epoch pointed to by reference antenna
        offset_azel = offset.azel(time, location)
        # enu vector pointing from reference antenna to offset point
        towards_pole = np.array(azel_to_enu(offset_azel.az.rad, offset_azel.alt.rad))
        # Obtain direction vector(s) from reference antenna to target
        azel = self.azel(time, location)
        # w axis points toward target
        w_basis = np.array(azel_to_enu(azel.az.rad, azel.alt.rad))
        # u axis is orthogonal to z and w
        u_basis = np.cross(towards_pole, w_basis, axis=0) * offset_sign
        u_basis /= np.linalg.norm(u_basis, axis=0)
        # v axis completes the orthonormal basis
        v_basis = np.cross(w_basis, u_basis, axis=0)
        return np.array([u_basis, v_basis, w_basis])

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
        antenna2 : :class:`~astropy.coordinates.EarthLocation` or
            :class:`Antenna` or sequence
            Second antenna of baseline pair (baseline vector points toward it), shape A
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equiv, optional
            Timestamp(s) of shape T, defaults to now
        antenna : :class:`~astropy.coordinates.EarthLocation` or
            :class:`Antenna`, optional
            First (reference) antenna of baseline pair, which also serves as
            pointing reference (defaults to default antenna)

        Returns
        -------
        uvw : :class:`~astropy.coordinates.CartesianRepresentation`, shape T + A
            (u, v, w) coordinates of baseline as Cartesian (x, y, z), in units
            of length. The shape is a concatenation of the `timestamp` and
            `antenna2` shapes.

        Notes
        -----
        All calculations are done in the local ENU coordinate system centered on
        the first antenna, as opposed to the traditional XYZ coordinate system.
        This avoids having to convert (az, el) angles to (ha, dec) angles and
        uses linear algebra throughout instead.
        """
        # Obtain basis vectors
        basis = self.uvw_basis(timestamp, antenna)
        antenna = self._valid_antenna(antenna)
        # Obtain baseline vector from reference antenna to second antenna(s)
        try:
            baseline = np.stack([antenna.baseline_toward(a2).xyz for a2 in antenna2])
        except TypeError:
            baseline = antenna.baseline_toward(antenna2).xyz
        # Apply linear coordinate transformation. A single call np.dot won't
        # work for both the scalar and array case, so we explicitly specify the
        # axes to sum over.
        return CartesianRepresentation(np.tensordot(basis, baseline, ([1], [-1])))

    def lmn(self, ra, dec, timestamp=None, antenna=None):
        """Calculate (l, m, n) coordinates for another target relative to self.

        Calculate (l, m, n) coordinates for another target, while pointing at
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
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equiv, optional
            Timestamp(s), defaults to now
        antenna : :class:`~astropy.coordinates.EarthLocation` or
            :class:`Antenna`, optional
            Pointing reference (defaults to default antenna)

        Returns
        -------
        l,m,n : float, or array of same length as `ra`, `dec`, `timestamps`
            (l, m, n) coordinates of target(s).
        """
        ref_radec = self.radec(timestamp, antenna)
        return sphere_to_ortho(ref_radec.ra.rad, ref_radec.dec.rad, ra, dec)

    @u.quantity_input(equivalencies=u.spectral())
    def flux_density(self, frequency: u.Hz = None) -> u.Jy:
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
        frequency : :class:`~astropy.units.Quantity`, optional
            Frequency at which to evaluate flux density

        Returns
        -------
        flux_density : :class:`~astropy.units.Quantity`
            Flux density in Jy, or np.nan if frequency is out of range or target
            does not have flux model. The shape matches the input.

        Raises
        ------
        ValueError
            If no frequency is specified, and no default frequency was set either
        """
        if frequency is None:
            frequency = self._flux_frequency
        if frequency is None:
            raise ValueError(
                "Please specify frequency at which to measure flux density"
            )
        if self.flux_model is None:
            # Target has no specified flux density
            return np.full(np.shape(frequency), np.nan) * u.Jy
        return self.flux_model.flux_density(frequency)

    @u.quantity_input(equivalencies=u.spectral())
    def flux_density_stokes(self, frequency: u.Hz = None) -> u.Jy:
        """Calculate flux density for given observation frequency(-ies), full-Stokes.

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
        frequency : :class:`~astropy.units.Quantity`, optional
            Frequency at which to evaluate flux density

        Returns
        -------
        flux_density : :class:`~astropy.units.Quantity`
            Flux density in Jy, or np.nan if frequency is out of range or target
            does not have flux model. The shape matches the input with an extra
            trailing dimension of size 4 containing Stokes I, Q, U, V.

        Raises
        ------
        ValueError
            If no frequency is specified, and no default frequency was set either
        """
        if frequency is None:
            frequency = self._flux_frequency
        if frequency is None:
            raise ValueError(
                "Please specify frequency at which to measure flux density"
            )
        if self.flux_model is None:
            return np.full(np.shape(frequency) + (4,), np.nan) * u.Jy
        return self.flux_model.flux_density_stokes(frequency)

    def separation(self, other_target, timestamp=None, antenna=None):
        """Angular separation between this target and another as viewed from antenna.

        Parameters
        ----------
        other_target : :class:`Target`
            The other target
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equiv, optional
            Timestamp(s) when separation is measured (defaults to now)
        antenna : :class:`~astropy.coordinates.EarthLocation` or
            :class:`Antenna`, optional
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
        time, location = self._astropy_funnel(timestamp, antenna)
        this_azel = self.azel(time, location)
        other_azel = other_target.azel(time, location)
        return this_azel.separation(other_azel)

    def sphere_to_plane(
        self,
        az,
        el,
        timestamp=None,
        antenna=None,
        projection_type="ARC",
        coord_system="azel",
    ):
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
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equiv, optional
            Timestamp(s), defaults to now
        antenna : :class:`~astropy.coordinates.EarthLocation` or
            :class:`Antenna`, optional
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
        if coord_system == "radec":
            # The target (ra, dec) coordinates will serve as reference point on sphere
            ref_radec = self.radec(timestamp, antenna)
            return sphere_to_plane[projection_type](
                ref_radec.ra.rad, ref_radec.dec.rad, az, el
            )
        else:
            # The target (az, el) coordinates will serve as reference point on sphere
            ref_azel = self.azel(timestamp, antenna)
            return sphere_to_plane[projection_type](
                ref_azel.az.rad, ref_azel.alt.rad, az, el
            )

    def plane_to_sphere(
        self,
        x,
        y,
        timestamp=None,
        antenna=None,
        projection_type="ARC",
        coord_system="azel",
    ):
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
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equiv, optional
            Timestamp(s), defaults to now
        antenna : :class:`~astropy.coordinates.EarthLocation` or
            :class:`Antenna`, optional
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
        if coord_system == "radec":
            # The target (ra, dec) coordinates will serve as reference point on sphere
            ref_radec = self.radec(timestamp, antenna)
            return plane_to_sphere[projection_type](
                ref_radec.ra.rad, ref_radec.dec.rad, x, y
            )
        else:
            # The target (az, el) coordinates will serve as reference point on sphere
            ref_azel = self.azel(timestamp, antenna)
            return plane_to_sphere[projection_type](
                ref_azel.az.rad, ref_azel.alt.rad, x, y
            )


def construct_azel_target(az, el):
    """Create unnamed stationary target (*azel* body type) **DEPRECATED**."""
    warnings.warn(
        "This function is deprecated and will be removed - "
        "use Target.from_azel(az, el) instead",
        FutureWarning,
    )
    return Target.from_azel(az, el)


def construct_radec_target(ra, dec):
    """Create unnamed fixed target (*radec* body type) **DEPRECATED**."""
    warnings.warn(
        "This function is deprecated and will be removed - "
        "use Target.from_radec(ra, dec) instead",
        FutureWarning,
    )
    return Target.from_radec(ra, dec)
