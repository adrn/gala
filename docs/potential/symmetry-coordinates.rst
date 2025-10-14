.. _potential-symmetry-coordinates:

*************************************
Using Symmetry Coordinates
*************************************

.. currentmodule:: gala.potential

Many gravitational potentials have special symmetries that make certain
coordinate systems more natural than Cartesian coordinates. For example,
spherically-symmetric potentials only depend on the radius :math:`r`, and
axisymmetric potentials only depend on the cylindrical radius :math:`R` and
height :math:`z`.

Gala provides support for **symmetry coordinates** that allow you to evaluate
potential methods using these natural coordinate systems, making your code
cleaner and more expressive.

Spherical Potentials
=====================

For spherically-symmetric potentials, you can use the ``r=`` keyword argument
to pass just the spherical radius instead of full 3D Cartesian coordinates.

Supported Spherical Potentials
-------------------------------

The following built-in potentials support spherical symmetry coordinates:

- :class:`~gala.potential.HernquistPotential`
- :class:`~gala.potential.PlummerPotential`
- :class:`~gala.potential.KeplerPotential`
- :class:`~gala.potential.IsochronePotential`
- :class:`~gala.potential.JaffePotential`
- :class:`~gala.potential.BurkertPotential`
- :class:`~gala.potential.SphericalSplinePotential`
- :class:`~gala.potential.NFWPotential` (when spherical: ``a=b=c=1``)

Examples
--------

Basic usage with scalar radius::

    >>> import astropy.units as u
    >>> import gala.potential as gp
    >>> from gala.units import galactic
    >>> pot = gp.HernquistPotential(m=1e10 * u.Msun, c=5 * u.kpc, units=galactic)
    >>> pot.energy(r=10 * u.kpc)  # doctest: +SKIP
    <Quantity -0.01792115 kpc2 / Myr2>

Arrays of radii::

    >>> import numpy as np
    >>> r = np.array([1.0, 5.0, 10.0, 50.0]) * u.kpc
    >>> pot.energy(r=r)  # doctest: +SKIP
    <Quantity [-0.06674208, -0.03367003, -0.01792115, -0.00396825] kpc2 / Myr2>

All potential methods support symmetry coordinates::

    >>> pot.gradient(r=r)  # Returns gradient in Cartesian coords  # doctest: +SKIP
    >>> pot.density(r=r)  # doctest: +SKIP
    >>> pot.acceleration(r=r)  # doctest: +SKIP
    >>> pot.mass_enclosed(r=r)  # doctest: +SKIP
    >>> pot.circular_velocity(r=r)  # doctest: +SKIP

Computing a mass profile::

    >>> r_profile = np.logspace(-1, 2, 100) * u.kpc
    >>> m_profile = pot.mass_enclosed(r=r_profile)  # doctest: +SKIP

This is much cleaner than the traditional approach::

    >>> # Old way (still works!)
    >>> pos = np.zeros((3, 100)) * u.kpc
    >>> pos[0] = r_profile
    >>> m_profile = pot.mass_enclosed(pos)  # doctest: +SKIP

Cylindrical (Axisymmetric) Potentials
======================================

For axisymmetric potentials, you can use ``R=`` and ``z=`` keyword arguments
to specify cylindrical coordinates.

Supported Cylindrical Potentials
---------------------------------

The following built-in potentials support cylindrical symmetry coordinates:

- :class:`~gala.potential.MiyamotoNagaiPotential`
- :class:`~gala.potential.CylSplinePotential`

Examples
--------

Basic usage with both R and z::

    >>> pot = gp.MiyamotoNagaiPotential(m=1e11 * u.Msun, a=3 * u.kpc, b=0.3 * u.kpc, units=galactic)
    >>> pot.energy(R=8 * u.kpc, z=0.5 * u.kpc)  # doctest: +SKIP

Arrays of coordinates::

    >>> R = np.linspace(4, 12, 100) * u.kpc
    >>> z = np.linspace(-1, 1, 100) * u.kpc
    >>> pot.energy(R=R, z=z)  # doctest: +SKIP

The ``z`` coordinate defaults to zero, making midplane calculations particularly
convenient::

    >>> # Evaluate in the midplane (z=0)
    >>> pot.energy(R=R)  # doctest: +SKIP
    >>> pot.circular_velocity(R=R)  # doctest: +SKIP

This is especially useful for plotting rotation curves::

    >>> R_curve = np.linspace(0.1, 20, 200) * u.kpc
    >>> v_circ = pot.circular_velocity(R=R_curve)  # Automatically uses z=0  # doctest: +SKIP

Broadcasting
------------

Symmetry coordinates support NumPy broadcasting. For example, to compute the
potential on a grid of (R, z) values::

    >>> R_grid = np.linspace(1, 15, 50) * u.kpc
    >>> z_grid = np.linspace(-3, 3, 30) * u.kpc
    >>> R_mesh, z_mesh = np.meshgrid(R_grid, z_grid)
    >>> phi = pot.energy(R=R_mesh, z=z_mesh)  # doctest: +SKIP
    >>> # phi has shape (30, 50)

Important Notes
===============

Return Coordinates
------------------

**Gradients and accelerations are always returned in Cartesian coordinates**,
even when using symmetry coordinate inputs. This design choice ensures:

1. **Physical consistency**: Forces and accelerations are vectors in 3D space
2. **Integration compatibility**: Orbit integration requires Cartesian coordinates
3. **Composability**: Results can be easily combined with other potentials

For example::

    >>> grad = pot.gradient(r=10 * u.kpc)  # doctest: +SKIP
    >>> # Returns shape (3, 1) array: [dx, dy, dz] in Cartesian coords
    >>> # with values [grad_x, 0, 0] since the input corresponds to x=10, y=0, z=0

Coordinate Validation
---------------------

Symmetry coordinates are validated to ensure they match the potential's
symmetry. Trying to use the wrong coordinates will raise a helpful error::

    >>> pot = gp.HernquistPotential(m=1e10 * u.Msun, c=5 * u.kpc, units=galactic)
    >>> pot.energy(R=10 * u.kpc)  # doctest: +SKIP
    ValueError: This potential has spherical symmetry and expects coordinate 'r',
    but you provided: {'R'}

You also cannot mix Cartesian and symmetry coordinates::

    >>> pos = [10, 0, 0] * u.kpc
    >>> pot.energy(pos, r=10 * u.kpc)  # doctest: +SKIP
    ValueError: Cannot specify both position `q` and symmetry coordinates

Keyword-Only Arguments
----------------------

Symmetry coordinates must be passed as keyword arguments, not positional
arguments. This makes the API clear and prevents ambiguity::

    >>> # Correct
    >>> pot.energy(r=10 * u.kpc)  # doctest: +SKIP

    >>> # Wrong - this passes a Cartesian position
    >>> pot.energy(10 * u.kpc)  # doctest: +SKIP

Implementing Custom Potentials with Symmetry
=============================================

If you're creating a custom potential class and want to add symmetry coordinate
support, set the ``_symmetry`` class attribute:

For a spherical potential::

    from gala.potential import PotentialBase
    from gala.potential.potential.symmetry import SphericalSymmetry

    class MySphericalPotential(PotentialBase):
        _symmetry = SphericalSymmetry()

        # ... rest of implementation

For a cylindrical potential::

    from gala.potential.potential.symmetry import CylindricalSymmetry

    class MyCylindricalPotential(PotentialBase):
        _symmetry = CylindricalSymmetry()

        # ... rest of implementation

The base class will automatically handle the coordinate transformation in all
potential methods.

See Also
========

- :ref:`Potential documentation <gala-potential>`
- :ref:`Defining custom potentials <define-new-potential>`
- :class:`~gala.potential.potential.symmetry.PotentialSymmetry`
- :class:`~gala.potential.potential.symmetry.SphericalSymmetry`
- :class:`~gala.potential.potential.symmetry.CylindricalSymmetry`
