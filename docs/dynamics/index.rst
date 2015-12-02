.. include:: ../references.txt

.. _gary-dynamics:

********************************
Dynamics (`gary.dynamics`)
********************************

Introduction
============

This subpackage contains functions and classes useful for gravitational
dynamics. The fundamental objects used by many of the functions and utilities
in this and other subpackages are the `~gary.dynamics.PhaseSpacePosition` and
`~gary.dynamics.Orbit` subclasses.

There are utilities for transforming orbits in phase-space to action-angle
coordinates, tools for visualizing and computing dynamical quantities from
orbits, and tools useful for nonlinear dynamics such as Lyapunov exponent
estimation.

For code blocks below and any pages linked below, I assume the following
imports have already been excuted::

    >>> import astropy.units as u
    >>> import numpy as np
    >>> import gary.potential as gp
    >>> import gary.dynamics as gd
    >>> from gary.units import galactic

Getting started: Working with orbits
====================================

Some simple tools are provided for inspecting and plotting orbits. For example,
we'll start by integrating an orbit in Cartesian coordinates using the
:mod:`gary.potential` and :mod:`gary.integrate` subpackages::

    >>> pot = gp.MiyamotoNagaiPotential(m=2.5E11, a=6.5, b=0.26, units=galactic)
    >>> w0 = gd.CartesianPhaseSpacePosition(pos=[11., 0., 0.2]*u.kpc,
    ...                                     vel=[0., 200, 100]*u.km/u.s)
    >>> orbit = pot.integrate_orbit(w0, dt=1., nsteps=1000)

This will integrate an orbit from the specified initial conditions (``w0``) and
return an orbit object. There are many useful methods of the `~gary.dynamics.Orbit`
subclasses and many functions that accept `~gary.dynamics.Orbit` objects. For example,
we can easily visualize the orbit by plotting the time series in all projections
using the :meth:`~gary.dynamics.CartesianOrbit.plot` method::

    >>> fig = orbit.plot()

.. plot::
    :align: center

    import astropy.units as u
    import gary.potential as gp
    import gary.dynamics as gd
    from gary.units import galactic
    pot = gp.MiyamotoNagaiPotential(m=2.5E11, a=6.5, b=0.26, units=galactic)
    w0 = gd.CartesianPhaseSpacePosition(pos=[11., 0., 0.2]*u.kpc,
                                        vel=[0., 200, 100]*u.km/u.s)
    orbit = pot.integrate_orbit(w0, dt=1., nsteps=1000)
    fig = orbit.plot()

From this object, we can easily compute dynamical quantities such as the energy
or angular momentum (I take the 0th element because these functions return the
quantities computed at every timestep)::

    >>> orbit.energy()[0] # doctest: +FLOAT_CMP
    <Quantity -0.06074019848886105 kpc2 / Myr2>

Let's see how well the integrator conserves energy and the ``z`` component of
angular momentum::

    >>> E = orbit.energy()
    >>> Lz = orbit.angular_momentum()[2]
    >>> np.std(E), np.std(Lz)
    (<Quantity 4.654233175716351e-06 kpc2 / Myr2>,
     <Quantity 9.675900603446092e-16 kpc2 / Myr>)

Using gary.dynamics
===================
More details are provided in the linked pages below:

.. toctree::
   :maxdepth: 1

   orbits-in-detail
   actionangle
   nonlinear
   orbitfit

.. automodapi:: gary.dynamics
