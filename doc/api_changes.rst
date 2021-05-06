API Changes for 1.0
===================

Behaviour changes
-----------------

Description strings
~~~~~~~~~~~~~~~~~~~

Astropy Quantities
~~~~~~~~~~~~~~~~~~

Initialisation from Astropy objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Catalogue filter intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~

Catalogue filters based on numerical values (like flux density, azimuth,
elevation and distance to a target) now use half-open intervals instead of
closed intervals. The lower limit is included by the filter while the upper
limit is not. For example, given a ``katpoint.Catalogue`` object ``cat``, the
filtered catalogue ``cat.filter(el_limit=[10, 30] * u.deg)`` has targets with
elevation angles in the range 10 <= el < 30 degrees in version 1.0, while the
filtered range used to be 10 <= el <= 30 degrees in version 0.x. This has
little practical impact, but does make it easier to partition a catalogue
into non-overlapping ranges.

Deprecations
------------

*add_stars* and *add_specials* parameters to Catalogue constructor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These parameters cannot be set to True anymore, and setting them to False
elicits a ``FutureWarning`` since they are slated for removal. Add these
targets manually to the Catalogue instead.

Removals
--------

Removed behaviour
~~~~~~~~~~~~~~~~~

- There is no more *star* body type; use *radec* instead.

Modules
~~~~~~~

- The entire ``katpoint.stars`` module has been removed. Load the EDB file in
  ``scripts/ephem_stars.edb`` into a Catalogue instead.
- The ``delay`` module has been split into the ``delay_model`` and
  ``delay_correction`` modules.

Classes, methods and attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``katpoint.DelayCorrection.extra_delay`` has been renamed to
  ``katpoint.DelayCorrection.extra_correction``.
- ``katpoint.Antenna.format_katcp`` has been removed; KATCP uses ``str()`` instead.
- ``katpoint.Target.format_katcp`` has been removed; KATCP uses ``str()`` instead.

Functions
~~~~~~~~~

Arguments
~~~~~~~~~

Development changes
-------------------

Increase to minimum supported versions of Python and dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Katpoint 1.0 depends on Python 3.6+ and Astropy 4.1+, while katpoint 0.x depends
on Python 2.7+ and PyEphem.

Use pytest instead of nose
~~~~~~~~~~~~~~~~~~~~~~~~~~
