1.10.0 (2025-07-31)
===================

New Features
------------

- Added a new ``SimulationUnitSystem`` class for handling unit systems in
  simulations, especially for N-body simulations.

- Added options ``error_if_fail`` and ``log_output`` to integrator kwargs for the
  dop853 integrator, along with some other arguments that are passed directly to the C
  integrator (e.g., ``nstiff``). ``error_if_fail`` controls whether Python will raise
  an error if the C integrator fails to integrate an orbit, and ``log_output`` will log
  the output of the integrator (primarily for errors) to stdout. See the docstring for `
  ``gala.integrate.DOP853Integrator`` for more information about all of the available
  options for the integrator.

- You may now specify a ``gala.units.UnitSystem`` instance to control the units of
  plotted components when using ``gala.dynamics.Orbit.plot()`` or
  ``gala.dynamics.PhaseSpacePosition.plot()``.

- Added the ability to specify integer or string (i.e. non-Quantity) potential
  parameters.

- Added ``gala.potential.EXPPotential`` for using basis function expansion potentials
  from EXP.

- Added methods ``NFWPotential.M200()``, ``NFWPotential.R200()``,
  ``NFWPotential.c200()`` to compute the characteristic mass, radius, and concentration
  of an NFW instance.

Bug fixes
---------

- Fixed a longstanding issue with orbit integration where there was a maximum number of
  orbits that could be integrated simultaneously. Now, arrays are allocated dynamically
  and there is no limit.

- Similarly, fixed a longstanding issue that restricted the number of potential
  components that could be added to a composite potential. Now, arrays are allocated
  dynamically and there is no limit.

- Some versions of Agama do not accept astropy.units objects as input to setUnits. Gala
  now converts to floats to set the unit scales in agama when converting a potential to
  Agama (using ``potential.as_interop("agama")``).

- Fixed a bug in ``MockStreamGenerator.run()`` where passing an array of length 1 for
  the progenitor mass would lead to a silent failure of the stream generation.

- Fixed the normalization of the ``PowerLawCutoffPotential`` potential energy so that it
  goes to zero at infinity.

API changes
-----------

- Gala has ``save_all`` and ``store_all`` flags for saving all orbits at every
  timestep. The ``store_all`` flag is now deprecated and will be removed in a future
  release. The ``save_all`` flag should be used instead.

Other
-----

- Added a flag to skip rotating and/or shifting input coordinates when computing
  potential, density, gradient, and hessian values. This leads to some free performance
  improvements in existing code!

- Refactored the way integration is done with the DOP853 integrator. The integrator now
  uses the dense output feature (which uses interpolation) to compute the output values
  at the requested times. This is a significant performance improvement for large
  numbers of orbits, and also allows for much faster results when integrating over long
  timescales.

1.9.1 (2024-08-26)
==================

- This release fixes the wheel builds for linux and mac and no new features or bug fixes
  are included.


1.9.0 (2024-08-22)
==================

New Features
------------

- Added an option to specify a multiprocessing or parallel processing pool when
  computing basis function coefficients for the SCF potential from discrete particles.

- Added the Burkert potential as a built-in cpotential.

- Added a method to generate the Burkert potential with just r0 as an input

- Added new particle spray method by Chen et al. (2024).

Bug fixes
---------

- Fixed the parameter values in the ``FardalStreamDF`` class to be consistent with
  the values used in Fardal et al. (2015). Added an option ``gala_modified`` to the
  class to enable using the new (correct) parameter values, but the default will
  continue to use the Gala modified values (for backwards compatibility).

- Improved internal efficiency of ``DirectNBody``.

- Fixed a bug in which passing a ``DirectNBody`` instance to the ``MockStreamGenerator.
  run()`` would fail if ``save_all=False`` in the nbody instance.

- Fixed an incompatibility with Astropy v6.1 and above where ``_make_getter`` was
  removed.


API changes
-----------

- Deprecated ``gala.integrate.Integrator.run`` for
  ``gala.integrate.Integrator.__call__``. The old method will raise a warning
  and will be removed in a future release.


1.8.1 (2023-12-31)
==================

- New release to fix upload to PyPI from GitHub Actions and invalid pin in pyia
  dependency.


1.8 (2023-12-23)
================

New Features
------------

- Added a ``.guiding_center()`` method to ``PhaseSpacePosition`` and ``Orbit`` to
  compute the guiding center radius.

- Added a way to convert Gala potential instances to Agama potential instances.

Bug fixes
---------

- Fixed a bug with the ``plot_contours()`` and ``plot_density_contours()`` methods so
  that times specified are now passed through correctly to the potential methods.

- Fixed the YAML output to use ``default_flow_style=None`` for serializing potential
  objects, which leads to a more efficient array output.

- ``scf.compute_coeffs_discrete`` now raises an error if GSL is not enabled rather than
  silently returning zeros

- ``SCFPotential`` will now work with IO functions (``save`` & ``load``)

- Fixes compatibility with Astropy v6.0

API changes
-----------

- Changed the way potential interoperability is done with other Galactic dynamics
  packages (Agama, galpy, etc.). It is now handled by the ``Potential.as_interop()``
  method on all potential class instances.


1.7.1 (2023-08-05)
==================

- Switched build system to use pyproject.toml instead of setup.cfg

1.7 (2023-08-05)
================

New Features
------------

- Added a method to export the internal components of an
  ``MN3ExponentialDiskPotential()`` to three ``MiyamotoNagaiPotential`` instances.

- Added a new Milky Way potential model: ``MilkyWayPotential2022``, which is based on
  updated measurements of the disk structure and circular velocity curve of the disk.

- Added the ability to use leapfrog integration within the ``DirectNBody`` integrator.

- Added a new coordinate frame for the Vasiliev+2021 Sagittarius stream coordinate
  system, ``SagittariusVasiliev21``.

Bug fixes
---------

- Fixed a bug with the ``OrphanKoposov19()`` coordinate frame that caused the wrong
  rotation matrix to be returned.

- Fixed an ``AstropyDeprecationWarning`` resulting from the use of ``override__dir__``.

- Fixed a bug in ``Orbit.estimate_period()`` that would cause the method to fail with a
  ``UnitsError`` if one orbit returned a nan value for the period.

- Fixed a bug when compiling the ``dop853`` integrator.

API changes
-----------

- Refactored the way ``GreatCircleICRSFrame()`` works to be more consistent and
  unambiguous with coordinate frame definitions. The frame now requires an input pole
  and origin, but can be initialized in old ways using the ``from_*()`` class methods
  (e.g., with ``pole`` and ``ra0`` values).


1.6.1 (2022-11-07)
==================

Bug fixes
---------

- Properly incorporate commits related to ``SCFInterpolatedPotential``.


1.6 (2022-11-07)
================

New Features
------------

- Added a ``.replicate()`` method to Potential classes to enable copying
  potential objects but modifying some parameter values.

- Added a new potential class ``MN3ExponentialDiskPotential`` based on Smith et
  al. (2015): an approximation of the potential generated by a double
  exponential disk using a sum of three Miyamoto-Nagai disks.

- The ``Orbit.estimate_period()`` method now returns period estimates in all
  phase-space components instead of just the radial period.

- Added a ``store_all`` flag to the integrators to control whether to save
  phase-space information for all timesteps or only the final timestep.

- Added a ``plot_rotation_curve()`` method to all potential objects to make a 1D plot
  of the circular velocity curve.

- Added a new potential for representing multipole expansions ``MultipolePotential``.

- Added a new potential ``CylSplinePotential`` for flexible representation of
  axisymmetric potentials by allowing passing in grids of potential values
  evaluated grids of R, z values (like the ``CylSpline`` potential in Agama).

- Added a ``show_time`` flag to ``Orbit.animate()`` to control whether to show the
  current timestep.

- Changed ``Orbit.animate()`` to allow for different ``marker_style`` and
  ``segment_style`` options for individual orbits by passing a list of dicts instead
  of just a dict.

- Added an experimental new class ``SCFInterpolatedPotential`` that accepts a time
  series of coefficients and interpolates the coefficient values to any evaluation time.

Bug fixes
---------

- Fixed a bug where the ``NFWPotential`` energy was nan when evaluating at the
  origin, and added tests for all potentials to check for a finite value of the
  potential at the origin (when expected).

- Fixed a bug in ``NFWPotential.from_M200_c()`` where the incorrect scale radius
  was computed (Cython does not always use Python 3 division rules for dividing
  integers!).

- Fixed a bug in the (C-level/internal) estimation of the 2nd derivative of the
  potential, used to generate mock streams, that affects non-conservative force
  fields.

API changes
-----------

- The ``Orbit.estimate_period()`` method now returns period estimates in all
  phase-space components instead of just the radial period.


1.5 (2022-03-03)
================

New Features
------------

- Implemented a basic progress bar for integrating orbits and mock streams. Pass
  ``progress=True`` with ``Integrator_kwargs`` when calling
  ``.integrate_orbit()``, or pass ``progress=True`` to
  ``MockStreamGenerator.run()``.

- Added a new symplectic integrator: The Ruth 4th-order integrator, implemented
  with the class ``Ruth4Integrator``.

- Added a ``Orbit.animate()`` method to make ``matplotlib`` animations of
  orbits.

- Modified ``Orbit._max_helper()`` to use a parabola instead of interpolation

- Added functionality to transform from action-angle coordinates to Cartesian
  position velocity coordinates in the Isochrone potential:
  ``gala.dynamics.actionangle.isochrone_aa_to_xv()``.

- Added a new method on ``DirectNBody`` to enable computing the instantaneous,
  mutual, N-body acceleration vectors ``DirectNBody.acceleration()``.

Bug fixes
---------

- Fixed ``find_actions()`` to accept an ``Orbit`` instance with multiple orbits.

- Fixed a bug that appeared when trying to release all mock stream particles at
  the same timestep (e.g., pericenter).

- Fixed a bug where time arrays returned from ``parse_time_specification``
  could come back with a non-float64 dtype.

- Fixed a bug with ``DirectNBody`` with composite potentials where only the
  first potential component would move as a body / particle.

- Fixed a bug with the Python implementation of Leapfrog integration
  ``LeapfrogIntegrator`` that led to incorrect orbits for non-conservative
  systems that were integrated backwards (i.e. with ``dt<<0``).

- Fixed a bug with the ``FlattenedNFW`` potential class in which the energy and
  gradient functions were not using the inputted flattening (``c`` value) and
  were instead defaulting to the spherical NFW model.

- Enabled pickling ``Frame`` instances and therefore now ``Hamiltonian``
  instances.

- Fixed a bug with ``autolim=True`` during Orbit plotting where the axes limits
  were only dependent on the most recent Orbit rather than all that were present
  on the axis

API changes
-----------

- Renamed ``gala.dynamics.actionangle.isochrone_to_aa()`` to
  ``gala.dynamics.actionangle.isochrone_xv_to_aa()``

- Renamed ``gala.dynamics.actionangle.find_actions()`` to
  ``gala.dynamics.actionangle.find_actions_o2gf()``


1.4.1 (2021-07-01)
==================

- Fixed a RST bug that caused the README to fail to render.


1.4 (2021-07-01)
================

New Features
------------

- ``UnitSystem`` objects can now be created with custom units passed in as
  Astropy ``Quantity`` objects.

- Added functionality to convert Gala potential objects to Galpy potential
  objects, or to create Gala potential objects from a pre-existing Galpy
  potential.

- Added a ``plot_3d()`` method for ``Orbit`` objects to make 3D plots of the
  orbital trajectories.

Bug fixes
---------

- Fixed a bug when calling ``orbit.norbits`` when the representation is not
  cartesian.

- Fixed a bug with ``GreatCircleICRSFrame.from_endpoints()`` that caused an
  error when the input coordinates had associated velocity data.

- Fixed a bug with the ``JaffePotential`` density evaluation, which was too low
  by a factor of two.

- Implemented a density function for ``LogarithmicPotential``, which was
  missing previously.

- The analytic action-angle and ``find_actions()`` utilities now correctly
  return frequencies with angular frequency units rather than frequency.

API changes
-----------

- Removed the deprecated ``gala.coordinates.get_galactocentric2019()`` function.


1.3 (2020-10-27)
================

New Features
------------

- Added a new ``.to_sympy()`` classmethod for the ``Potential`` classes to
  return a sympy expression and variables.

- Added a method, ``.to_galpy_orbit()``, to convert Gala ``Orbit`` instances to
  Galpy ``Orbit`` objects.

- The ``NFWPotential`` can now be instantiated via a new classmethod:
  ``NFWPotential.from_M200_c()``, which accepts a virial mass and a
  concentration.

- Added a fast way of computing the Staeckel focal length, ``Delta``, using
  Gala potential classes, ``gala.dynamics.get_staeckel_fudge_delta``

Bug fixes
---------

- Fixed a bug with ``Potential`` classes ``.replace_units()`` so that classes
  with dimensionless unit systems cannot be replaced with physical unit systems,
  and vice versa.

- Implemented Hessian functions for most potentials.

- Fixed ``.to_latex()`` to properly return a latex representation of the
  potential. This uses the new ``.to_sympy()`` method under the hood.

- Potential classes now validate that input positions have dimensionality that
  matches what is expected for each potential.

API changes
-----------

- Changed the way new ``Potential`` classes are defined: they now rely on
  defining class-level ``PotentialParameter`` objects, which reduces a
  significant amount of boilerplate code in the built-in potentials.


1.2 (2020-07-13)
================

- Gala now builds on Windows!

New Features
------------

- Added a coordinate frame for the Pal 13 stream, ``Pal13Shipp20``.

Bug fixes
---------

- Fixed a bug with the mock stream machinery in which the stream would not
  integrate for the specified number of timesteps if an array of
  ``n_particles`` was passed in with 0's near the end of the array.


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
- Fixed a bug in the ``OrphanKoposov19`` transformation.

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
