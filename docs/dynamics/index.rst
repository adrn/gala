.. _dynamics:

********************************
Dynamics (`gary.dynamics`)
********************************

Introduction
============

This subpackage contains functions and classes useful for advanced gravitational
dynamics. Much of the code is focused on transforming orbits in phase-space to
either action-angle coordinates or frequency-space, but there are other useful
tools for computing Lyapunov exponents and classifying orbits.

Orbits and phase-space positions are typically stored as Numpy :class:`~numpy.ndarray` objects with the convention that the *last* axis -- `axis=-1` -- is the phase-
space dimensionality. For example, for a collection of 100, 3D cartesian positions
(x,y,z), this would be represented as an array with shape `(100,3)`. Or, for orbits
of 100 particles over 1000 timesteps for the full phase-space (including velocities),
the array would have shape `(1000,100,6)`.

Getting started
===============



For a detailed example that makes use of the code for transforming to
action-angle coordinates, see: :ref:`actionangle`.


General
=======

.. autosummary::
   :nosignatures:
   :toctree: _dynamics/

   gary.dynamics.angular_momentum
   gary.dynamics.classify_orbit

Angle-action
============

.. autosummary::
   :nosignatures:
   :toctree: _dynamics/

   gary.dynamics.find_actions
   gary.dynamics.isochrone_xv_to_aa
   gary.dynamics.isochrone_aa_to_xv
   gary.dynamics.harmonic_oscillator_xv_to_aa
   gary.dynamics.harmonic_oscillator_aa_to_xv

Misc. / Utility
===============

.. autosummary::
   :nosignatures:
   :toctree: _dynamics/

   gary.dynamics.generate_n_vectors
   gary.dynamics.unwrap_angles
   gary.dynamics.fit_isochrone
   gary.dynamics.fit_harmonic_oscillator
