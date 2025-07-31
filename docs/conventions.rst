
.. _conventions:

***********
Conventions
***********

.. _name-conventions:

Common variable names
=====================

This package uses standard variable names throughout for consistency:

- ``w`` represents phase-space coordinates (positions and velocities)
- ``q`` represents positions only
- ``p`` or ``v`` represent velocities or momenta
- ``t`` represents time arrays

.. _shape-conventions:

Array shapes
============

Arrays and :class:`~astropy.units.Quantity` objects in ``Gala`` follow
consistent shape conventions:

**Coordinate arrays**: ``axis=0`` is the coordinate dimension. For example,
128 different 3D Cartesian positions have shape ``(3, 128)``.

**Orbit collections**: Arrays have three axes:
- ``axis=0``: coordinate dimension
- ``axis=1``: time axis
- ``axis=2``: different orbits

.. _energy-momentum:

Energy and momentum
===================

In `gala`, energy and angular momentum quantities are *per unit mass* unless
otherwise specified. This applies to:

- Potential energy
- Kinetic energy
- Total energy
- Angular momentum
- Linear momentum
- Conjugate momenta
