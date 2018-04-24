0.3 (2018-04-23)
================

New Features
------------

- Added a ``NullPotential`` class that has 0 mass and serves as a placeholder.
- Added a new ``zmax()`` method on the ``Orbit`` class to compute the maximum z
  heights and times, or the mean maximum z height. Similar to ``apocenter()``
  and ``pericenter()``.
- Added a new generator method on the ``Orbit`` class for easy iteration over
  orbits.

Bug fixes
---------

- ``Orbit.norbits`` now works...oops.
- ``apocenter()`` and ``pericenter()`` now work when more than one orbit is
  stored in an ``Orbit`` class.

0.2.2 (2017-10-07)
==================

New features
------------
- Added a new coordinate frame aligned with the Palomar 5 stream.
- Added a function ``gala.dynamics.combine`` to combine ``PhaseSpacePosition``
  or ``Orbit`` objects.

Bug fixes
---------
- Added a density function for the Kepler potential.
- Added a density function for the Long & Murali bar potential

Other changes
-------------
- Added JOSS paper.
- Cleaned up some tests and documentation to use the ``Hamiltonian`` object.

0.2.1 (2017-07-19)
==================

Bug fixes
---------
- Array parameters are now numpy.ravel'd before being passed to the
  ``CPotentialWrapper`` class methods.
- Added attribution to Bovy 2015 for disk potential in MilkyWayPotential

0.2 (2017-07-15)
================

New Features
------------
- Added a new potential class for the Satoh density (Satoh 1980).
- Added support for Leapfrog integration when generating mock stellar streams.
- Added new colormaps and defaults for the matplotlib style.
- Added support for non-inertial reference frames and implemented a constant
  rotating reference frame.
- Added a new class - ``Hamiltonian`` - for storing potentials with reference
  frames. This should be used for easy orbit integration instead of the
  potential classes.
- Added a new argument to the mock stream generation functions t output orbits
  of all of the mock stream star particles to an HDF5 file.
- Cleaned up and simplified the process of subclassing a C-implemented
  gravitational potential.
- Gravitational potential class instances can now be composed by just adding the
  instances.
- Added a ``MilkyWayPotential`` class.

API-breaking changes
--------------------
- ``CartesianPhaseSpacePosition`` and ``CartesianOrbit`` are deprecated. Use
  ``PhaseSpacePosition`` and ``Orbit`` with a Cartesian representation instead.
- Overhauled the storage of position and velocity information on
  ``PhaseSpacePosition`` and ``Orbit`` classes. This uses new features in
  Astropy 2.0 that allow attaching "differential" classes to representation
  classes for storing velocity information. ``.pos`` and ``.vel`` no longer
  point to arrays of Cartesian coordinates, but now instead point to
  astropy.coordinates representation and differential objects, respectively.

0.1.1 (2016-05-20)
==================

- Removed debug statement.
- Added 'Why' page to documentation.

0.1.0 (2016-05-19)
==================

- Initial release.
