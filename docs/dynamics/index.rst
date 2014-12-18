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


Reference/API
=============

.. autosummary::
   :nosignatures:
   :toctree: _dynamics/

   gary.dynamics.angular_momentum
   gary.dynamics.classify_orbit
   gary.dynamics.find_actions
