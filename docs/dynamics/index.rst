.. include:: ../references.txt

.. _gary-dynamics:

********************************
Dynamics (`gary.dynamics`)
********************************

Introduction
============

This subpackage contains functions and classes useful for advanced gravitational
dynamics. Much of the code is focused on transforming orbits in phase-space to
either action-angle coordinates or frequency-space. There are other useful
tools for computing Lyapunov exponents and classifying orbits.

Getting started
===============

Conventions
-----------

Orbits and phase-space positions are typically stored as Numpy :class:`~numpy.ndarray` objects with the convention that the *last* axis (``axis=-1``) is the phase-
space dimensionality. For example, for a collection of 100, 3D cartesian positions
(x,y,z), this would be represented as an array with shape ``(100,3)``. Or, for orbits
of 100 particles over 1000 timesteps for the full phase-space (including velocities),
the array would have shape ``(1000,100,6)``.

Analyzing orbits
----------------

Some simple tools are provided for inspecting and plotting orbits. For example,
imagine we have an orbit in Cartesian coordinates (by integrating an orbit using the
:ref:`gary.potential` and :ref:`gary.integrate` submodules)::

   >>> from gary.units import galactic
   >>> import gary.potential as gp
   >>> p = gp.MiyamotoNagaiPotential(m=2.5E11, a=6.5, b=0.26, units=galactic)
   >>> w0 = [11., 0., 0.2, 0., 0.2, 0., -0.025]
   >>> t,w = p.integrate_orbit(w0, dt=1., nsteps=10000)

We may want to quickly plot the orbit in all projections to get an idea of the
geometry of the orbit. We can use the `gd.plot_orbits` function to generate a
3-panel plot of the orbit in all projections (x-y, x-z, y-z)::

   >>> fig = gd.plot_orbits(w, marker=None, linestyle='-')


Transforming to angle-action coordinates
----------------------------------------


Tutorial
========

For a detailed example that makes use of the code for transforming to
action-angle coordinates, see: :ref:`actionangle`.

API
===

.. automodapi:: gary.dynamics
