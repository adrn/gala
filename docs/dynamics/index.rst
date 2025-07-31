.. include:: references.txt

.. _gala-dynamics:

********************************
Dynamics (`gala.dynamics`)
********************************

For the examples below the following imports have already been executed::

    >>> import astropy.units as u
    >>> import numpy as np
    >>> import gala.potential as gp
    >>> import gala.dynamics as gd
    >>> from gala.units import galactic

Introduction
============

This subpackage contains functions and classes useful for gravitational
dynamics. There are utilities for transforming orbits in phase-space to
action-angle coordinates, tools for visualizing and computing dynamical
quantities from orbits, tools to generate mock stellar streams, and tools useful
for nonlinear dynamics such as Lyapunov exponent estimation.

The fundamental objects used by many of the functions and utilities in this and
other subpackages are the |psp| and |orb| classes.

Getting started: Working with orbits
====================================

We'll demonstrate the |psp| and |orb| objects by first integrating an orbit::

    >>> pot = gp.MiyamotoNagaiPotential(
    ...     m=2.5e11 * u.Msun, a=6.5 * u.kpc, b=0.26 * u.kpc, units=galactic
    ... )
    >>> w0 = gd.PhaseSpacePosition(
    ...     pos=[11.0, 0.0, 0.2] * u.kpc, vel=[0.0, 200, 100] * u.km / u.s
    ... )
    >>> orbit = gp.Hamiltonian(pot).integrate_orbit(w0, dt=1.0, n_steps=1000)

This numerically integrates an orbit from the specified initial conditions,
``w0``, and returns an |orb| object. By default, the position and velocity are
assumed to be Cartesian coordinates but other coordinate systems are supported
(see the :ref:`orbits-in-detail` and :ref:`nd-representations` pages for more
information).

The |orb| object that is returned contains many useful methods, and can be
passed to many of the analysis functions implemented in Gala. For example, we
can easily visualize the orbit by plotting the time series in all Cartesian
projections using the :meth:`~gala.dynamics.Orbit.plot` method::

    >>> fig = orbit.plot()

.. plot::
    :align: center

    import astropy.units as u
    import gala.potential as gp
    import gala.dynamics as gd
    from gala.units import galactic
    pot = gp.MiyamotoNagaiPotential(m=2.5e11, a=6.5, b=0.26, units=galactic)
    w0 = gd.PhaseSpacePosition(
        pos=[11.0, 0.0, 0.2] * u.kpc, vel=[0.0, 200, 100] * u.km / u.s
    )
    orbit = gp.Hamiltonian(pot).integrate_orbit(w0, dt=1.0, n_steps=1000)
    fig = orbit.plot()

We can also visualize the orbit in transformed coordinates, for example,
cylindrical radius :math:`\rho` and :math:`z`::

    >>> fig = orbit.represent_as("cylindrical").plot(["rho", "z"])

.. plot::
    :align: center
    :width: 60%

    import astropy.units as u
    import gala.potential as gp
    import gala.dynamics as gd
    from gala.units import galactic
    pot = gp.MiyamotoNagaiPotential(m=2.5e11, a=6.5, b=0.26, units=galactic)
    w0 = gd.PhaseSpacePosition(
        pos=[11.0, 0.0, 0.2] * u.kpc, vel=[0.0, 200, 100] * u.km / u.s
    )
    orbit = gp.Hamiltonian(pot).integrate_orbit(w0, dt=1.0, n_steps=1000)
    _ = orbit.represent_as("cylindrical").plot(["rho", "z"])

The |orb| object also enables computing dynamical quantities such as
energy or angular momentum::

    >>> E = orbit.energy()
    >>> E[0] # doctest: +SKIP
    <Quantity −0.060740198 kpc2 / Myr2>

Let's check how well the integrator conserves energy and the ``z`` component of
angular momentum::

    >>> Lz = orbit.angular_momentum()[2]
    >>> np.std(E), np.std(Lz) # doctest: +FLOAT_CMP
    (<Quantity 4.654233175716351e-06 kpc2 / Myr2>,
     <Quantity 9.675900603446092e-16 kpc2 / Myr>)

We can access the position and velocity components of the orbit separately using
attributes that map to the underlying `~astropy.coordinates.BaseRepresentation`
and `~astropy.coordinates.BaseDifferential` subclass instances that store the
position and velocity data. The attribute names depend on the representation.
For example, for a Cartesian representation, the position components are ``["x",
"y", "z"]`` and the velocity components are ``["v_x", "v_y", "v_z"]``. With a
|orb| or |psp| instance, you can check the valid compnent names using the
attributes ``.pos_components`` and ``.vel_components``::

    >>> orbit.pos_components.keys()  # doctest: +SKIP
    odict_keys(["x", "y", "z"])
    >>> orbit.vel_components.keys()  # doctest: +SKIP
    odict_keys(["v_x", "v_y", "v_z"])

Meaning, we can access these components by doing, e.g.::

    >>> orbit.v_x  # doctest: +FLOAT_CMP
    <Quantity [ 0.        , -0.00567589, -0.01129934, ...,  0.18751756,
                0.18286687,  0.17812762] kpc / Myr>

For a Cylindrical representation, these are instead::

    >>> cyl_orbit = orbit.represent_as("cylindrical")
    >>> cyl_orbit.pos_components.keys()  # doctest: +SKIP
    odict_keys(["rho", "phi", "z"])
    >>> cyl_orbit.vel_components.keys()  # doctest: +SKIP
    odict_keys(["v_rho", "pm_phi", "v_z"])
    >>> cyl_orbit.v_rho  # doctest: +FLOAT_CMP
    <Quantity [ 0.        , -0.00187214, -0.00369183, ...,  0.01699321,
                0.01930216,  0.02159477] kpc / Myr>

Continue to the :ref:`orbits-in-detail` page for more information.

Using gala.dynamics
===================

More details are provided in the linked pages below:

.. toctree::
   :maxdepth: 2

   orbits-in-detail
   nd-representations
   actionangle
   mockstreams
   nonlinear
   nbody


API
===

.. automodapi:: gala.dynamics
    :include: PhaseSpacePosition
    :include: Orbit
    :no-inheritance-diagram:
