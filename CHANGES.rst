1.1 (2020-03-08)
================

New Features
------------
- Potential objects now support replacing the unit system with the
  ``.replace_units()`` method, or by updating the ``.units`` attribute on an
  existing instance.
- Added a ``DirectNBody`` class that supports direct N-body orbit integration in
  (optional) external background potentials.
- Added a coordinate frame for the Jhelum stream, ``JhelumBonaca19``.
- Added a function for retrieving a more modern Galactocentric reference frame,
  ``gala.coordinates.get_galactocentric2019()``.
- Added a classmethod to allow initializing a ``GreatCircleICRSFrame`` from a
  rotation matrix that specifies the transformation from ``ICRS`` to the great
  circle frame.

Bug fixes
---------
- Fixed an issue that led to incorrect ``GreatCircleICRSFrame`` transformations
  when no ``ra0`` was provided.
- Fixed a bug in the ``OrphanKoposov19`` tranasformation.

API changes
-----------
- Overhauled the mock stellar stream generation methodology to allow for more
  general stream generation. See ``MockStreamGenerator`` and the stream
  distribution function classes, e.g., ``FardalStreamDF``.
- Removed deprecated ``CartesianPhaseSpacePosition`` class.
- Removed long deprecated ``Quaternion`` class.


1.0 (2019-04-12)
================

New Features
------------
- Added a new coordinate frame for great circle coordinate systems defined by a
  pole. This frame can be created with a pole and origin, a pole and longitude
  zero point, by two points along a great circle, or by specifying the cartesian
  basis vectors of the new frame.
- Added a function to transform a proper motion covariance matrix to a new
  coordinate frame.
- Added support for compiling Gala with or without the GNU Scientific Library
  (GSL), which is needed for the new potential classes indicated below.
- Added a new ``PowerLawCutoffPotential`` class for a power-law density
  distribution with an exponential cutoff *(requires GSL)*.
- Added an implementation of the ``MWPotential2014`` from ``galpy`` (called
  ``BovyMWPotential2014`` in ``gala``) *(requires GSL)*.
- Added an implementation of the Self-Consistent Field (SCF) basis function
  expansion method for representing potential-density pairs *(requires GSL)*.
- Most Potential classes now support rotations and origin shifts through the
  ``R`` and ``origin`` arguments.
- Added a ``progress`` argument to the Python integrators to display a progress
  bar when stepping the integrators.
- When generating mock stellar streams and storing snapshots (rather than just
  the final phase-space positions of the particles) now supports specifying the
  snapshot frequency with the ``output_every`` argument.

Bug fixes
---------
- Stream frames now properly wrap the longitude (``phi1``) components to the
  range (-180, 180) deg.

API changes
-----------
- Stream classes have been renamed to reflect the author that defined them.
- Proper motion and coordinate velocity transformations have now been removed in
  favor of the implementations in Astropy.
- Added a ``.data`` attribute to ``PhaseSpacePosition`` objects that returns a
  ``Representation`` or ``NDRepresentation`` instance with velocity data
  (differentials) attached.

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
