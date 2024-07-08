################################################################################
# Copyright (c) 2009-2011,2013,2016-2023, National Research Foundation (SARAO)
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

"""Target catalogue."""

import logging
import warnings
from collections import defaultdict

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle, Longitude
from astropy.time import Time

from .target import Target
from .timestamp import Timestamp

logger = logging.getLogger(__name__)

specials = [
    "Sun",
    "Moon",
    "Mercury",
    "Venus",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
]


def _normalised(name):
    """Normalise string to make name lookup more robust."""
    return name.strip().lower().replace(" ", "").replace("_", "")


# -------------------------------------------------------------------------------------
# --- CLASS :  Catalogue
# -------------------------------------------------------------------------------------


class Catalogue:
    """A searchable and filterable catalogue of targets.

    Overview
    --------

    A :class:`Catalogue` object combines two concepts:

    - A list of targets, which can be filtered, sorted, pretty-printed and
      iterated over. The list is accessible as :meth:`Catalogue.targets`, and
      the catalogue itself is iterable, returning the next target on each
      iteration. The targets are assumed to be unique, but may have the same
      name. An example is::

        cat = katpoint.Catalogue()
        cat.add(some_targets)
        t = cat.targets[0]
        for t in cat:
            # Do something with target t

    - Lookup by name, by using the catalogue as if it were a dictionary. This
      is simpler for the user, who does not have to remember all the target
      details. The named lookup supports tab completion in IPython, which
      further simplifies finding a target in the catalogue. The most recently
      added target with the specified name is returned. An example is::

        targets = ['Sun, special', 'Moon, special', 'Jupiter, special']
        cat = katpoint.Catalogue(targets)
        t = cat['Sun']

    Construction
    ------------

    A catalogue can be constructed in many ways. The simplest way is::

        cat = katpoint.Catalogue()

    which produces an empty catalogue. Additional targets may be loaded during
    initialisation of the catalogue by providing a list of :class:`Target`
    objects (or a single object by itself), as in the following example::

        t1 = katpoint.Target('Venus, special')
        t2 = katpoint.Target('Takreem, azel, 20, 30')
        cat1 = katpoint.Catalogue(t1)
        cat2 = katpoint.Catalogue([t1, t2])

    Alternatively, the list of targets may be replaced by a list of target
    description strings (or a single description string). The target objects
    are then constructed before being added, as in::

        cat1 = katpoint.Catalogue('Takreem, azel, 20, 30')
        cat2 = katpoint.Catalogue(['Venus, special', 'Takreem, azel, 20, 30'])

    Taking this one step further, the list may be replaced by any iterable
    object that returns strings. A very useful example of such an object is the
    Python :class:`file` object, which iterates over the lines of a text file.
    If the catalogue file contains one target description string per line
    (with comments and blank lines allowed too), it may be loaded as::

        cat = katpoint.Catalogue(open('catalogue.csv'))

    Once a catalogue is initialised, more targets may be added to it. The
    :meth:`Catalogue.add` method is the most direct way. It accepts a single
    target object, a list of target objects, a single string, a list of strings
    or a string iterable. This is illustrated below::

        t1 = katpoint.Target('Venus, special')
        t2 = katpoint.Target('Takreem, azel, 20, 30')
        cat = katpoint.Catalogue()
        cat.add(t1)
        cat.add([t1, t2])
        cat.add('Venus, special')
        cat.add(['Venus, special', 'Takreem, azel, 20, 30'])
        cat.add(open('catalogue.csv'))

    Some target types are typically found in files with standard formats.
    Notably, *tle* targets are found in TLE files with three lines per target,
    and many *xephem* targets are stored in EDB database files. Editing these
    files to make each line a valid :class:`Target` description string is
    cumbersome, especially in the case of TLE files which are regularly updated.
    Two special methods simplify the loading of targets from these files::

        cat = katpoint.Catalogue()
        cat.add_tle(open('gps-ops.txt'))
        cat.add_edb(open('hipparcos.edb'))

    Whenever targets are added to the catalogue, a tag or list of tags may be
    specified. The tags can also be given as a single string of
    whitespace-delimited tags, since tags may not contain whitespace. These tags
    are added to the targets currently being added. This makes it easy to tag
    groups of related targets in the catalogue, as shown below::

        cat = katpoint.Catalogue(tags='default')
        cat.add_tle(open('gps-ops.txt'), tags='gps satellite')
        cat.add_tle(open('glo-ops.txt'), tags=['glonass', 'satellite'])
        cat.add(open('source_list.csv'), tags='calibrator')
        cat.add_edb(open('hipparcos.edb'), tags='star')

    Finally, targets may be removed from the catalogue. The most recently added
    target with the specified name is removed from the targets list as well as
    the lookup dict. The target may be removed via any of its names::

        targets = ['Sol | Sun, special', 'Moon, special', 'Jupiter, special']
        cat = katpoint.Catalogue(targets)
        cat.remove('Sun')

    Filtering and sorting
    ---------------------

    A :class:`Catalogue` object may be filtered based on various criteria. The
    following filters are available:

    - *Tag filter*. Returns all targets that have a specified set of tags, and
      *not* another set of tags. Tags prepended with a tilde (~) indicate tags
      which targets should not have. All tags have to be present (or absent) for
      a target to be selected. Remember that the body type is also a tag. An
      example is::

        cat = katpoint.Catalogue(tags='default')
        cat1 = cat.filter(tags=['special', '~radec'])
        cat1 = cat.filter(tags='special ~radec')

    - *Flux filter*. Returns all targets with a flux density between the
      specified limits, at a given frequency. If only one limit is given, it is
      a lower limit. To simplify filtering, a default flux frequency may be
      supplied to the catalogue during initialisation. This is stored in each
      target in the catalogue. An example is::

        import astropy.units as u
        cat = katpoint.Catalogue(open('source_list.csv'))
        cat1 = cat.filter(flux_limit=[1, 100] * u.Jy,
                          flux_frequency=1500 * u.MHz)
        cat = katpoint.Catalogue(open('source_list.csv'),
                                 flux_frequency=1500 * u.MHz)
        cat1 = cat.filter(flux_limit=1 * u.Jy)

    - *Azimuth filter*. Returns all targets with an azimuth angle in the given
      range. The range is specified in degrees as [left, right], where *left* is
      the leftmost or starting azimuth, and *right* is the rightmost or ending
      azimuth. The azimuth angle increases clockwise from *left* to *right* to
      form the range. If *right* is less than *left*, the azimuth angles range
      around +-180 degrees. Since the target azimuth is dependent on time and
      observer position, a timestamp and :class:`katpoint.Antenna` object has to
      be provided. The timestamp defaults to now, and the antenna object may be
      associated with the catalogue during initialisation, from where it is
      stored in each target. An example is::

        import astropy.units as u
        ant = katpoint.Antenna('XDM, -25:53:23, 27:41:03, 1406, 15.0')
        targets = ['Sun, special', 'Moon, special', 'Jupiter, special']
        cat = katpoint.Catalogue(targets)
        cat1 = cat.filter(az_limit=[0, 90] * u.deg,
                          timestamp='2009-10-10', antenna=ant)
        cat = katpoint.Catalogue(antenna=ant)
        cat1 = cat.filter(az_limit=[90, 0] * u.deg)

    - *Elevation filter*. Returns all targets with an elevation angle within the
      given limits, in degrees. If only one limit is given, it is assumed to be
      a lower limit. As with the azimuth filter, a timestamp and antenna object
      is required (or defaults will be used). An example is::

        import astropy.units as u
        ant = katpoint.Antenna('XDM, -25:53:23, 27:41:03, 1406, 15.0')
        targets = ['Sun, special', 'Moon, special', 'Jupiter, special']
        cat = katpoint.Catalogue(targets)
        cat1 = cat.filter(el_limit=[10, 30] * u.deg,
                          timestamp='2009-10-10', antenna=ant)
        cat = katpoint.Catalogue(antenna=ant)
        cat1 = cat.filter(el_limit=10 * u.deg)

    - *Proximity filter*. Returns all targets with angular separation from a
      given set of targets within a specified range. The range is given as a
      lower and upper limit, in degrees, and a single number is taken as the
      lower limit. The typical use of this filter is to return all targets more
      than a specified number of degrees away from a known set of interfering
      targets. As with the azimuth filter, a timestamp and antenna object is
      required (or defaults will be used). An example is::

        import astropy.units as u
        ant = katpoint.Antenna('XDM, -25:53:23, 27:41:03, 1406, 15.0')
        targets = ['Sun, special', 'Moon, special', 'Jupiter, special']
        cat = katpoint.Catalogue(targets)
        cat.add_tle(open('geo.txt'))
        sun = cat['Sun']
        afristar = cat['AFRISTAR']
        cat1 = cat.filter(dist_limit=5 * u.deg,
                          proximity_targets=[sun, afristar],
                          timestamp='2009-10-10', antenna=ant)
        cat = katpoint.Catalogue(antenna=ant)
        cat1 = cat.filter(dist_limit=[0, 5] * u.deg, proximity_targets=sun)

    The criteria may be divided into *static* criteria which are independent of
    time (tags and flux) and *dynamic* criteria which do depend on time
    (azimuth, elevation and proximity). There are two filtering mechanisms that
    both support the same criteria, but differ on their handling of dynamic
    criteria:

    - A direct filter, implemented by the :meth:`Catalogue.filter` method. This
      returns the filtered catalogue as a new catalogue which contains the
      subset of targets that satisfy the criteria. All criteria are evaluated at
      the same time instant. A typical use-case is::

        cat = katpoint.Catalogue(open('source_list.csv'))
        strong_sources = cat.filter(flux_limit=10 * u.Jy,
                                    flux_frequency=1500 * u.MHz)

    - An iterator filter, implemented by the :meth:`Catalogue.iterfilter`
      method. This is a Python *generator function*, which returns a
      *generator iterator*, to be more precise. Each time the returned
      iterator's .next() method is invoked, the next suitable :class:`Target`
      object is returned. If no timestamp is provided, the criteria are
      re-evaluated at the time instant of the .next() call, which makes it easy
      to cycle through a list of targets over an extended period of time (as
      during observation). The iterator filter is typically used in a for-loop::

        cat = katpoint.Catalogue(open('source_list.csv'))
        ant = katpoint.Antenna('XDM, -25:53:23, 27:41:03, 1406, 15.0')
        for t in cat.iterfilter(el_limit=10 * u.deg, antenna=ant):
            # < observe target t >

    When a catalogue is sorted, the order of the target list is changed. The
    catalogue may be sorted according to name (the default), right ascension,
    declination, azimuth, elevation and flux. Any position-based key requires a
    timestamp and :class:`katpoint.Antenna` object to evaluate the position of
    each target, and the flux key requires a frequency at which to evaluate the
    flux.

    Parameters
    ----------
    targets : :class:`Target` object or string, or sequence of these, optional
        Target or list of targets to add to catalogue (may also be file object)
    tags : string or sequence of strings, optional
        Tag or list of tags to add to *targets* (strings will be split on
        whitespace)
    add_specials: bool, optional
        Always False (add special bodies yourself) **DEPRECATED**
    add_stars:  bool, optional
        Always False (stars have no special support anymore) **DEPRECATED**
    antenna : :class:`Antenna` object, optional
        Default antenna to use for position calculations for all targets
    flux_frequency : :class:`~astropy.units.Quantity`, optional
        Default frequency at which to evaluate flux density of all targets

    Notes
    -----
    The catalogue object has an interesting relationship with orderedness.
    While it is nominally an ordered list of targets, it is considered equal to
    another catalogue with the same targets in a different order. This is
    because the catalogue may be conveniently reordered in many ways (e.g.
    based on elevation, declination, flux, etc.) while remaining essentially
    the *same* catalogue. It also allows us to preserve the order in which the
    catalogue was assembled, which seems the most natural.
    """

    @u.quantity_input(equivalencies=u.spectral())
    def __init__(
        self,
        targets=None,
        tags=None,
        add_specials=None,
        add_stars=None,
        antenna=None,
        flux_frequency: u.Hz = None,
    ):
        self.lookup = defaultdict(list)
        self.targets = []
        self._antenna = antenna
        self._flux_frequency = flux_frequency
        if add_specials is not None:
            if add_specials:
                raise ValueError(
                    "The add_specials parameter is not supported anymore - "
                    "please add the targets manually"
                )
            warnings.warn(
                "The add_specials parameter is now permanently False "
                "and will be removed",
                FutureWarning,
            )
        if add_stars is not None:
            if add_stars:
                raise ValueError(
                    "The add_stars parameter is not supported anymore - "
                    "please add the stars manually (see scripts/ephem_stars.edb)"
                )
            warnings.warn(
                "The add_specials parameter is now permanently False "
                "and will be removed",
                FutureWarning,
            )
        if targets is None:
            targets = []
        self.add(targets, tags)

    # Provide properties to pass default antenna or flux frequency changes on to targets
    @property
    def antenna(self):
        """Default antenna used to calculate target positions."""
        return self._antenna

    @antenna.setter
    def antenna(self, ant):
        self._antenna = ant
        for target in self.targets:
            target.antenna = ant

    @property
    def flux_frequency(self):
        """Default frequency at which to evaluate flux density."""
        return self._flux_frequency

    @flux_frequency.setter
    def flux_frequency(self, frequency):
        self._flux_frequency = frequency
        for target in self.targets:
            target.flux_frequency = frequency

    def __str__(self):
        """Target description strings making up the catalogue, joined by newlines."""
        return "\n".join([str(target) for target in self.targets]) + "\n"

    def __repr__(self):
        """Short human-friendly string representation of catalogue object."""
        targets = len(self.targets)
        names = len(self.lookup.keys())
        return f"<katpoint.Catalogue targets={targets} names={names} at {id(self):#x}>"

    def __len__(self):
        """Return number of targets in catalogue."""
        return len(self.targets)

    def _targets_with_name(self, name):
        """Return list of targets in catalogue with given name (or alias)."""
        return self.lookup.get(_normalised(name), [])

    def __getitem__(self, name):
        """Look up target name in catalogue and return target object.

        This returns the most recently added target with the given name.
        The name string may be tab-completed in IPython to simplify finding
        a target.

        Parameters
        ----------
        name : string
            Target name to look up (can be alias as well)

        Returns
        -------
        target : :class:`Target` object, or None
            Associated target object, or None if no target was found
        """
        try:
            return self._targets_with_name(name)[-1]
        except IndexError:
            return None

    def __contains__(self, obj):
        """Test whether catalogue contains exact target, or target with given name."""
        if isinstance(obj, Target):
            return obj in self._targets_with_name(obj.name)
        else:
            return _normalised(obj) in self.lookup

    def __eq__(self, other):
        """Equality comparison operator (ignores order of targets)."""
        return isinstance(other, Catalogue) and set(self.targets) == set(other.targets)

    def __hash__(self):
        """Hash value is independent of order of targets in catalogue."""
        return hash(frozenset(self.targets))

    def __iter__(self):
        """Iterate over targets in catalogue."""
        return iter(self.targets)

    def _ipython_key_completions_(self):
        """List of keys used in IPython (version >= 5) tab completion."""
        names = set()
        for target in self.targets:
            names.add(target.name)
            for alias in target.aliases:
                names.add(alias)
        return sorted(names)

    def add(self, targets, tags=None):
        """Add targets to catalogue.

        Examples of catalogue construction can be found in the :class:`Catalogue`
        documentation.

        Parameters
        ----------
        targets : :class:`Target` object or string, or sequence of these
            Target or list of targets to add to catalogue (may also be file object)
        tags : string or sequence of strings, optional
            Tag or list of tags to add to *targets* (strings will be split on
            whitespace)

        Examples
        --------
        Here are some ways to add targets to a catalogue:

        >>> from katpoint import Catalogue
        >>> cat = Catalogue()
        >>> cat.add(open('source_list.csv'), tags='cal')
        >>> cat.add('Sun, special')
        >>> cat2 = Catalogue()
        >>> cat2.add(cat.targets)
        """
        if isinstance(targets, (Target, str)):
            targets = [targets]
        for target in targets:
            if isinstance(target, str):
                # Ignore strings starting with a hash (assumed to be comments)
                # or only containing whitespace
                if (len(target.strip()) == 0) or (target[0] == "#"):
                    continue
                target = Target(target)
            if not isinstance(target, Target):
                raise ValueError(
                    "List of targets should either contain "
                    "Target objects or description strings"
                )
            # Add tags first since they affect target identity / description
            target.add_tags(tags)
            if target in self:
                logger.warning(
                    "Skipped '%s' [%s] (already in catalogue)",
                    target.name,
                    target.tags[0],
                )
                continue
            existing_names = [name for name in target.names if name in self]
            if existing_names:
                logger.warning(
                    "Found different targets with same name(s) '%s' in catalogue",
                    ", ".join(existing_names),
                )
            target.antenna = self.antenna
            target.flux_frequency = self.flux_frequency
            self.targets.append(target)
            for name in target.names:
                self.lookup[_normalised(name)].append(target)
            logger.debug(
                "Added '%s' [%s] (and %d aliases)",
                target.name,
                target.tags[0],
                len(target.aliases),
            )

    def add_tle(self, lines, tags=None):
        r"""Add NORAD Two-Line Element (TLE) targets to catalogue.

        Examples of catalogue construction can be found in the :class:`Catalogue`
        documentation.

        Parameters
        ----------
        lines : sequence of strings
            List of lines containing one or more TLEs (may also be file object)
        tags : string or sequence of strings, optional
            Tag or list of tags to add to targets (strings will be split on
            whitespace)

        Examples
        --------
        Here are some ways to add TLE targets to a catalogue:

        >>> from katpoint import Catalogue
        >>> cat = Catalogue()
        >>> cat.add_tle(open('gps-ops.txt'), tags='gps')
        >>> lines = [
            'ISS DEB [TOOL BAG]\n',
            '1 33442U 98067BL  09195.86837279  .00241454  37518-4  34022-3 0  3424\n',
            '2 33442  51.6315 144.2681 0003376 120.1747 240.0135 16.05240536 37575\n'
            ]
        >>> cat2.add_tle(lines)
        """
        targets, tle = [], []
        for line in lines:
            if (line[0] == "#") or (len(line.strip()) == 0):
                continue
            tle += [line]
            if len(tle) == 3:
                name, line1, line2 = [raw_line.strip() for raw_line in tle]
                targets.append(Target(f"{name}, tle, {line1}, {line2}"))
                tle = []
        if len(tle) > 0:
            logger.warning(
                "Did not receive a multiple of three lines when constructing TLEs"
            )

        # Check TLE epochs and warn if some are too far in past or future,
        # which would make TLE inaccurate right now
        max_epoch_age = 0 * u.day
        num_outdated = 0
        worst = None
        for target in targets:
            # Use orbital period to distinguish near-earth and deep-space objects
            # (which have different accuracies)
            mean_motion = target.body.satellite.no_kozai * u.rad / u.minute
            orbital_period = 1 * u.cycle / mean_motion
            epoch_age = Time.now() - target.body.epoch
            direction = "past" if epoch_age > 0 * u.day else "future"
            epoch_age = abs(epoch_age)
            # Near-earth models should be good for about a week (conservative estimate)
            if orbital_period < 225 * u.minute and epoch_age > 7 * u.day:
                num_outdated += 1
                if epoch_age > max_epoch_age:
                    worst = (
                        f"Worst case: TLE epoch for '{target.name}' is "
                        f"{epoch_age.jd:.0f} days in the {direction}, "
                        "should be <= 7 for near-Earth model"
                    )
                    max_epoch_age = epoch_age
            # Deep-space models = more accurate (three weeks for conservative estimate)
            if orbital_period >= 225 * u.minute and epoch_age > 21 * u.day:
                num_outdated += 1
                if epoch_age > max_epoch_age:
                    worst = (
                        f"Worst case: TLE epoch for '{target.name}' is "
                        f"{epoch_age.jd:.0f} days in the {direction}, "
                        "should be <= 21 for deep-space model"
                    )
                    max_epoch_age = epoch_age
        if num_outdated > 0:
            logger.warning(
                "%d of %d TLE set(s) are outdated, probably making them inaccurate "
                "for use right now",
                num_outdated,
                len(targets),
            )
            logger.warning(worst)
        self.add(targets, tags)

    def add_edb(self, lines, tags=None):
        r"""Add XEphem database format (EDB) targets to catalogue.

        Examples of catalogue construction can be found in the :class:`Catalogue`
        documentation.

        Parameters
        ----------
        lines : sequence of strings
            List of lines containing a target per line (may also be file object)
        tags : string or sequence of strings, optional
            Tag or list of tags to add to targets (strings will be split on
            whitespace)

        Examples
        --------
        Here are some ways to add EDB targets to a catalogue:

        >>> from katpoint import Catalogue
        >>> cat = Catalogue()
        >>> cat.add_edb(open('hipparcos.edb'), tags='star')
        >>> lines = ['HYP71683,f|S|G2,14:39:35.88 ,-60:50:7.4 ,-0.010,2000,\n',
                     'HYP113368,f|S|A3,22:57:39.055,-29:37:20.10,1.166,2000,\n']
        >>> cat2.add_edb(lines)
        """
        targets = []
        for line in lines:
            if (line[0] == "#") or (len(line.strip()) == 0):
                continue
            targets.append("xephem," + line.replace(",", "~"))
        self.add(targets, tags)

    def remove(self, name):
        """Remove target from catalogue.

        This removes the most recently added target with the given name
        from the catalogue. If the target is not in the catalogue, do nothing.

        Parameters
        ----------
        name : string
            Name of target to remove (may also be an alternate name of target)
        """
        target = self[name]
        if target is not None:
            for name_to_scrap in target.names:
                targets_with_name = self.lookup[_normalised(name_to_scrap)]
                targets_with_name.remove(target)
                if not targets_with_name:
                    del self.lookup[_normalised(name_to_scrap)]
            self.targets.remove(target)

    def save(self, filename):
        """Save catalogue to file in CSV format.

        Parameters
        ----------
        filename : string
            Name of file to write catalogue to (overwriting existing contents)
        """
        with open(filename, "w", encoding="utf-8") as file:
            file.writelines([t.description + "\n" for t in self.targets])

    def closest_to(self, target, timestamp=None, antenna=None):
        """Determine target in catalogue that is closest to given target.

        The comparison is based on the apparent angular separation between the
        targets, as seen from the specified antenna and at the given time instant.

        Parameters
        ----------
        target : :class:`Target`
            Target with which catalogue targets are compared
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equiv, optional
            Timestamp at which to evaluate target positions (defaults to now)
        antenna : :class:`Antenna`, optional
            Antenna which points at targets (defaults to default antenna)

        Returns
        -------
        closest_target : :class:`Target` or None
            Target in catalogue that is closest to given *target*, or None if
            catalogue is empty
        min_dist : :class:`~astropy.coordinates.Angle`
            Angular separation between *target* and *closest_target*
        """
        if len(self.targets) == 0:
            return None, Angle(180.0 * u.deg)
        # Use the given target's default antenna unless it has none
        if antenna is None and target.antenna is None:
            antenna = self.antenna
        dist = np.stack(
            [target.separation(tgt, timestamp, antenna) for tgt in self.targets]
        )
        closest = dist.argmin()
        return self.targets[closest], dist[closest]

    _FILTER_PARAMETERS_DOCSTRING = """
        Parameters
        ----------
        tags : string, or sequence of strings, optional
            Tag or list of tags which targets should have. Tags prepended with
            a tilde (~) indicate tags which targets should *not* have. The string
            may contain multiple tags separated by whitespace. If None or an
            empty list, all tags are accepted. Remember that the body type is
            also a tag.
        flux_limit : :class:`~astropy.units.Quantity`, optional
            Allowed flux density range. If this is a single number, it is the
            lower limit, otherwise it takes the form [lower, upper]. If None,
            any flux density is accepted.
        flux_frequency : :class:`~astropy.units.Quantity`, optional
            Frequency at which to evaluate the flux density
        az_limit : :class:`~astropy.units.Quantity`, optional
            Allowed azimuth range. It takes the form [left, right], where *left*
            is the leftmost or starting azimuth, and *right* is the rightmost or
            ending azimuth. If *right* is less than *left*, the azimuth angles
            range around +-180. If None, any azimuth is accepted.
        el_limit : :class:`~astropy.units.Quantity`, optional
            Allowed elevation range. If this is a single number, it is the
            lower limit, otherwise it takes the form [lower, upper]. If None,
            any elevation is accepted.
        dist_limit : :class:`~astropy.units.Quantity`, optional
            Allowed range of angular distance to proximity targets. If this is
            a single number, it is the lower limit, otherwise it takes the form
            [lower, upper]. If None, any distance is accepted.
        proximity_targets : :class:`Target`, or sequence of :class:`Target`
            Target or list of targets used in proximity filter
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equiv, optional
            Timestamp at which to evaluate target positions (defaults to now).
            For :meth:`iterfilter` the default is the current time *at each iteration*.
        antenna : :class:`Antenna` object, optional
            Antenna which points at targets (defaults to default antenna)
        """.strip()

    @u.quantity_input(equivalencies=u.spectral())
    def iterfilter(
        self,
        tags=None,
        flux_limit: u.Jy = None,
        flux_frequency: u.Hz = None,
        az_limit: u.deg = None,
        el_limit: u.deg = None,
        dist_limit: u.deg = None,
        proximity_targets=None,
        timestamp=None,
        antenna=None,
    ):
        """Yield targets satisfying various criteria (generator function).

        This returns a (generator-)iterator which returns targets satisfying
        various criteria, one at a time. The standard use of this method is in a
        for-loop (i.e. ``for target in cat.iterfilter(...):``). This differs from
        the :meth:`filter` method in that all time-dependent criteria (such as
        elevation) may be evaluated at the time of the specific iteration, and
        not in advance as with :meth:`filter`. This simplifies finding the next
        suitable target during an extended observation of several targets.

        {Parameters}

        Returns
        -------
        iter : iterator object
            The generator-iterator object which will return filtered targets

        Raises
        ------
        ValueError
            If some required parameters are missing or limits are invalid

        Examples
        --------
        Here are some ways to filter a catalogue iteratively:

        >>> from katpoint import Catalogue, Antenna
        >>> import astropy.units as u
        >>> ant = Antenna('XDM, -25:53:23, 27:41:03, 1406, 15.0')
        >>> targets = ['Sun, special', 'Moon, special', 'Jupiter, special']
        >>> cat = Catalogue(targets, antenna=ant)
        >>> for t in cat.iterfilter(el_limit=10 * u.deg):
                # Observe target t
                pass
        """
        tag_filter = tags is not None
        flux_filter = flux_limit is not None
        azimuth_filter = az_limit is not None
        elevation_filter = el_limit is not None
        proximity_filter = dist_limit is not None
        # Copy targets to a new list which will be pruned by filters
        targets = list(self.targets)

        # First apply static criteria (tags, flux) which do not depend on timestamp
        if tag_filter:
            if isinstance(tags, str):
                tags = tags.split()
            desired_tags = {tag for tag in tags if tag[0] != "~"}
            undesired_tags = {tag[1:] for tag in tags if tag[0] == "~"}
            if desired_tags:
                targets = [
                    target for target in targets if set(target.tags) & desired_tags
                ]
            if undesired_tags:
                targets = [
                    target
                    for target in targets
                    if not set(target.tags) & undesired_tags
                ]

        if flux_filter:
            if flux_limit.isscalar:
                flux_limit = flux_limit, np.inf * u.Jy  # scalar becomes lower limit
            try:
                flux_lower, flux_upper = flux_limit
            except ValueError as err:
                raise ValueError(
                    "Flux limit should have the form <lower> or [<lower>, <upper>], "
                    f"not {flux_limit}"
                ) from err
            targets = [
                target
                for target in targets
                if flux_lower <= target.flux_density(flux_frequency) < flux_upper
            ]

        # Now prepare for dynamic criteria (azimuth, elevation, proximity)
        # which depend on potentially changing timestamp
        if azimuth_filter:
            try:
                # Wrap negative azimuth values to the expected range of 0-360 degrees
                az_left, az_right = Longitude(az_limit)
            except TypeError as err:
                raise ValueError(
                    "Azimuth limit should have the form [<left>, <right>], "
                    f"not {az_limit}"
                ) from err
        if elevation_filter:
            if el_limit.isscalar:
                el_limit = el_limit, None  # scalar becomes lower limit
            try:
                el_lower, el_upper = el_limit
            except ValueError as err:
                raise ValueError(
                    "Elevation limit should have the form <lower> "
                    f"or [<lower>, <upper>], not {el_limit}"
                ) from err
        if proximity_filter:
            if proximity_targets is None:
                raise ValueError(
                    "Please specify proximity target(s) for proximity filter"
                )
            if dist_limit.isscalar:
                dist_limit = dist_limit, None  # scalar becomes lower limit
            try:
                dist_lower, dist_upper = dist_limit
            except ValueError as err:
                raise ValueError(
                    "Distance limit should have the form <lower> "
                    f"or [<lower>, <upper>], not {dist_limit}"
                ) from err
            if isinstance(proximity_targets, Target):
                proximity_targets = [proximity_targets]

        # Keep checking targets while there are some in the list
        while targets:
            latest_timestamp = timestamp
            # Obtain current time if no timestamp is supplied.
            # This will differ for each iteration.
            if (
                azimuth_filter or elevation_filter or proximity_filter
            ) and latest_timestamp is None:
                latest_timestamp = Timestamp()
            # Iterate over targets until one is found that satisfies dynamic criteria
            for n, target in enumerate(targets):
                if azimuth_filter or elevation_filter:
                    azel = target.azel(latest_timestamp, antenna)
                if azimuth_filter:
                    if az_left <= az_right and not azel.az.is_within_bounds(
                        az_left, az_right
                    ):
                        continue
                    if az_left > az_right and azel.az.is_within_bounds(
                        az_right, az_left
                    ):
                        continue
                if elevation_filter:
                    if not azel.alt.is_within_bounds(el_lower, el_upper):
                        continue
                if proximity_filter:
                    dist = np.stack(
                        [
                            target.separation(prox_target, latest_timestamp, antenna)
                            for prox_target in proximity_targets
                        ]
                    )
                    if not dist.is_within_bounds(dist_lower, dist_upper):
                        continue
                # Break if target is found...
                # Popping the target inside the for-loop is a bad idea!
                found_one = n
                break
            else:
                # No targets in list satisfied dynamic criteria - iterator stops
                return
            # Return successful target and remove from list
            # to ensure it is not picked again.
            yield targets.pop(found_one)

    @u.quantity_input(equivalencies=u.spectral())
    def filter(
        self,
        tags=None,
        flux_limit: u.Jy = None,
        flux_frequency: u.Hz = None,
        az_limit: u.deg = None,
        el_limit: u.deg = None,
        dist_limit: u.deg = None,
        proximity_targets=None,
        timestamp=None,
        antenna=None,
    ):
        """Filter catalogue on various criteria.

        This returns a new catalogue containing the subset of targets that
        satisfy the given criteria. All criteria are evaluated at the same time
        instant. For real-time continuous filtering, consider using
        :meth:`iterfilter` instead.

        {Parameters}

        Returns
        -------
        subset : :class:`Catalogue`
            Filtered catalogue

        Raises
        ------
        ValueError
            If some required parameters are missing or limits are invalid

        Examples
        --------
        Here are some ways to filter a catalogue:

        >>> from katpoint import Catalogue, Antenna
        >>> import astropy.units as u
        >>> ant = Antenna('XDM, -25:53:23, 27:41:03, 1406, 15.0')
        >>> targets = ['Sun, special', 'Moon, special', 'Jupiter, special']
        >>> cat = Catalogue(targets, antenna=ant, flux_frequency=1500 * u.MHz)
        >>> cat1 = cat.filter(el_limit=10 * u.deg)
        >>> cat2 = cat.filter(az_limit=[150, -150] * u.deg)
        >>> cat3 = cat.filter(flux_limit=10 * u.Jy)
        >>> cat4 = cat.filter(tags='special ~radec')
        >>> cat5 = cat.filter(dist_limit=5 * u.deg, proximity_targets=cat['Sun'])
        """
        # Ensure that iterfilter operates on a single unique timestamp
        timestamp = Timestamp(timestamp)
        return Catalogue(
            list(
                self.iterfilter(
                    tags,
                    flux_limit,
                    flux_frequency,
                    az_limit,
                    el_limit,
                    dist_limit,
                    proximity_targets,
                    timestamp,
                    antenna,
                )
            ),
            antenna=self.antenna,
            flux_frequency=self.flux_frequency,
        )

    @u.quantity_input(equivalencies=u.spectral())
    def sort(
        self,
        key="name",
        ascending=True,
        flux_frequency: u.Hz = None,
        timestamp=None,
        antenna=None,
    ):
        """Sort targets in catalogue.

        This returns a new catalogue with the target list sorted according to
        the given key.

        Parameters
        ----------
        key : {'name', 'ra', 'dec', 'az', 'el', 'flux'}, optional
            Sort the targets according to this field
        ascending : {True, False}, optional
            True if key should be sorted in ascending order
        flux_frequency : :class:`~astropy.units.Quantity`, optional
            Frequency at which to evaluate the flux density
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equiv, optional
            Timestamp at which to evaluate target positions (defaults to now)
        antenna : :class:`Antenna` object, optional
            Antenna which points at targets (defaults to default antenna)

        Returns
        -------
        sorted : :class:`Catalogue` object
            Sorted catalogue

        Raises
        ------
        ValueError
            If some required parameters are missing or key is unknown
        """
        # Set up values that will be sorted
        if key == "name":
            values = [target.name for target in self.targets]
        elif key == "ra":
            values = [target.radec(timestamp, antenna).ra for target in self.targets]
        elif key == "dec":
            values = [target.radec(timestamp, antenna).dec for target in self.targets]
        elif key == "az":
            values = [target.azel(timestamp, antenna).az for target in self.targets]
        elif key == "el":
            values = [target.azel(timestamp, antenna).alt for target in self.targets]
        elif key == "flux":
            values = [target.flux_density(flux_frequency) for target in self.targets]
        else:
            raise ValueError("Unknown key to sort on")
        # Sort targets indirectly, either in ascending or descending order
        index = np.stack(values).argsort()
        if ascending:
            self.targets = np.array(self.targets, dtype=object)[index].tolist()
        else:
            self.targets = np.array(self.targets, dtype=object)[
                np.flipud(index)
            ].tolist()
        return self

    @u.quantity_input(equivalencies=u.spectral())
    def visibility_list(
        self, timestamp=None, antenna=None, flux_frequency: u.Hz = None, antenna2=None
    ):
        r"""Print out list of targets in catalogue, sorted by decreasing elevation.

        This prints out the name, azimuth and elevation of each target in the
        catalogue, in order of decreasing elevation. The motion of the target at
        the given timestamp is indicated by a character code, which is '/' if
        the target is rising, '\' if it is setting, and '-' if it is stationary
        (i.e. if the elevation angle changes by less than 1 arcminute during the
        one-minute interval surrounding the timestamp).

        The method indicates the horizon itself by a line of dashes. It also
        displays the target flux density if a frequency is supplied, and the
        delay and fringe period if a second antenna is supplied. It is useful
        to quickly see which targets are visible (or will be soon).

        Parameters
        ----------
        timestamp : :class:`~astropy.time.Time`, :class:`Timestamp` or equiv, optional
            Timestamp at which to evaluate target positions (defaults to now)
        antenna : :class:`Antenna`, optional
            Antenna which points at targets (defaults to default antenna)
        flux_frequency : :class:`~astropy.units.Quantity`, optional
            Frequency at which to evaluate flux density
        antenna2 : :class:`Antenna`, optional
            Second antenna of baseline pair (baseline vector points from
            *antenna* to *antenna2*), used to calculate delays and fringe rates
            per target
        """
        above_horizon = True
        timestamp = Timestamp(timestamp)
        if antenna is None:
            antenna = self.antenna
        if antenna is None:
            raise ValueError("Antenna object needed to calculate target position")
        title = f"Targets visible from antenna '{antenna.name}' at {timestamp.local()}"
        if flux_frequency is None:
            flux_frequency = self.flux_frequency
        if flux_frequency is not None:
            title += f", with flux density (Jy) evaluated at {flux_frequency:g}"
        if antenna2 is not None:
            title += (
                f" and fringe period (s) toward antenna "
                f"'{antenna2.name}' at same frequency"
            )
        print(title)
        print()
        print(
            "Target                        Azimuth    Elevation <    Flux Fringe period"
        )
        print(
            "------                        -------    --------- -    ---- -------------"
        )
        azels = [
            target.azel(timestamp + (-30.0, 0.0, 30.0), antenna)
            for target in self.targets
        ]
        elevations = np.stack([azel[1].alt for azel in azels])
        for index in np.argsort(elevations)[::-1]:
            target = self.targets[index]
            azel = azels[index][1]
            delta_el = azels[index][2].alt - azels[index][0].alt
            el_code = (
                "-"
                if (np.abs(delta_el) < 1 * u.arcmin)
                else ("/" if delta_el > 0 else "\\")
            )
            # If no flux frequency is given, do not attempt to evaluate the flux,
            # as it will fail.
            if flux_frequency is None:
                flux = np.nan
            else:
                flux = target.flux_density(flux_frequency).to_value(u.Jy)
            if antenna2 is not None and flux_frequency is not None:
                _, delay_rate = target.geometric_delay(antenna2, timestamp, antenna)
                if delay_rate != 0:
                    fringe_period = 1.0 / (delay_rate * flux_frequency.to_value(u.Hz))
                else:
                    fringe_period = np.inf
            else:
                fringe_period = None
            if above_horizon and azel.alt < 0.0:
                # Draw horizon line
                print(74 * "-")
                above_horizon = False
            az = azel.az.wrap_at(180 * u.deg).to_string(sep=":", precision=1)
            el = azel.alt.to_string(sep=":", precision=1)
            line = f"{target.name:<24s} {az:>12s} {el:>12s} {el_code:1s}"
            line = line + f" {flux:7.1f}" if not np.isnan(flux) else line + "        "
            if fringe_period is not None:
                line += f"    {fringe_period:10.2f}"
            print(line)

    iterfilter.__doc__ = iterfilter.__doc__.format(
        Parameters=_FILTER_PARAMETERS_DOCSTRING
    )
    filter.__doc__ = filter.__doc__.format(Parameters=_FILTER_PARAMETERS_DOCSTRING)
