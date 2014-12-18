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
