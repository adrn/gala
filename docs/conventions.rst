
.. _conventions:

***********
Conventions
***********

.. _name-conventions:

Common variable names
=====================

Unless otherwise stated (in function or class docstrings), this package tries to
adhere to using standard variable names for function and class arguments, and in
example code. For shorthand, the variable ``w`` is used to represented arrays of
phase-space coordinates (e.g., positions _and_ velocities). When representing
only positions, the variable ``q`` is used. For just velocities or momenta, the
variables ``p`` or ``v`` are used.

.. _shape-conventions:

Array shapes
============

The arrays and :class:`~astropy.units.Quantity` objects expected as input and
returned as ouput throughout ``Gary`` have shapes that follow a particular
convention, unless otherwise specified in function or class docstrings.

For arrays containing coordinate or kinematic information, ``axis=0`` is assumed
to be the coordinate dimension. For example, for representing 128 Cartesian 3D
positions, the object would have shape ``(3,128)``.

For collections of orbits, arrays have 3 axes. As above, ``axis=0`` is assumed
to be the coordinate dimension, but now ``axis=1`` is interpreted as the time
axis, and ``axis=2`` are the individual orbits.

.. _energy-momentum:

Energy and momentum
===================

The `gala` documentation and functions often refer to energy and angular
momentum and the respective quantities *per unit mass* interchangibly. Unless
otherwise specified, all such quantities -- e.g., energy, angular momentum,
momenta, conjugate momenta -- are in fact used and returned *per unit mass*.
