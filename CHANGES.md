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
