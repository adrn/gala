.. _potential-symmetry-coordinates:

*************************************
Using Symmetry Coordinates
*************************************

.. currentmodule:: gala.potential

Many gravitational potentials have symmetries that make certain coordinate systems more
natural than Cartesian coordinates. For example, spherically-symmetric potentials only
depend on the radius :math:`r`, and axisymmetric potentials only depend on the
cylindrical radius :math:`R` and height :math:`z`.

For potentials with these symmetries, you can use **symmetry coordinates** as a
shorthand instead of full 3D Cartesian coordinates. Internally, gala operates in
Cartesian coordinates, so the symmetry coordinates are simply a front-end convenience.

Spherical Potentials
=====================

For spherically-symmetric potentials, you can use the ``r=`` keyword argument to pass
just the spherical radius instead of full 3D Cartesian coordinates in many methods of
the potential classes.

Supported Spherical Potentials
-------------------------------

Any of the spherical potential models support using the ``r=...`` shorthand, including:

- :class:`~gala.potential.KeplerPotential`
- :class:`~gala.potential.HernquistPotential`
- :class:`~gala.potential.PlummerPotential`
- :class:`~gala.potential.SphericalSplinePotential`
- :class:`~gala.potential.NFWPotential` (when spherical: ``a=b=c=1``)

and more.

Examples
--------

Basic usage with a scalar radius value::

    >>> import astropy.units as u
    >>> import numpy as np
    >>> import gala.potential as gp
    >>> from gala.units import galactic
    >>> pot = gp.HernquistPotential(m=1e10 * u.Msun, c=5 * u.kpc, units=galactic)
    >>> pot.energy(r=10 * u.kpc)
    <Quantity [-0.002999] kpc2 / Myr2>

Arrays of radii::

    >>> r = [1.0, 5.0, 10.0, 50.0] * u.kpc
    >>> pot.energy(r=r)
    <Quantity [-0.0074975 , -0.0044985 , -0.002999  , -0.00081791] kpc2 / Myr2>

All potential methods support symmetry coordinates::

    >>> pot.gradient(r=r)  # Returns gradient in Cartesian coords  # doctest: +SKIP
    >>> pot.density(r=r)  # doctest: +SKIP
    >>> pot.acceleration(r=r)  # doctest: +SKIP
    >>> pot.mass_enclosed(r=r)  # doctest: +SKIP
    >>> pot.circular_velocity(r=r)  # doctest: +SKIP

Computing a mass profile::

    >>> r_profile = np.logspace(-1, 2, 100) * u.kpc
    >>> pot.mass_enclosed(r=r_profile)
    <Quantity [3.84467513e+06, 5.94521099e+06, 9.17160412e+06, 1.41075771e+07,
           ...] solMass>


This is much cleaner than the traditional approach::

    >>> # Explicit / old way with Cartesian arrays (still works!)
    >>> pos = np.zeros((3, 100)) * u.kpc
    >>> pos[0] = r_profile
    >>> m_profile = pot.mass_enclosed(pos)


Cylindrical (Axisymmetric) Potentials
======================================

For axisymmetric potentials, you can use ``R=`` and ``z=`` keyword arguments
to specify cylindrical coordinates in many methods on potential instances.

Supported Cylindrical Potentials
---------------------------------

The following built-in potentials support cylindrical symmetry coordinates:

- :class:`~gala.potential.MiyamotoNagaiPotential`
- :class:`~gala.potential.MN3ExponentialDiskPotential`
- :class:`~gala.potential.CylSplinePotential`

and others.


Examples
--------

Basic usage with both R and z::

    >>> pot = gp.MiyamotoNagaiPotential(m=1e11 * u.Msun, a=3 * u.kpc, b=0.3 * u.kpc, units=galactic)
    >>> pot.energy(R=8 * u.kpc, z=0.5 * u.kpc)
    <Quantity [-0.05131901] kpc2 / Myr2>

Arrays of coordinates::

    >>> R = np.linspace(4, 12, 32) * u.kpc
    >>> z = np.linspace(-1, 1, 32) * u.kpc
    >>> pot.energy(R=R, z=z)
    <Quantity [-0.07908656, -0.07715922, -0.07521451, -0.07326905, -0.07133667,
           ...] kpc2 / Myr2>

The ``z`` coordinate defaults to zero, making midplane calculations particularly
convenient::

    >>> # Evaluate in the midplane (z=0)
    >>> E = pot.energy(R=R)
    >>> pot.circular_velocity(R=R)  # assumes z=0
    <Quantity [222.1505711 , 223.33522621, 223.8929296 , 223.93564426,
           ...] km / s>



Composite Potentials
====================

:class:`~gala.potential.CompositePotential` objects automatically inherit symmetry from
their components based on some simple rules:

Symmetry Inheritance Rules:

- **All components spherical**: composite is spherical (can use ``r=``)
- **All components cylindrical**: composite is cylindrical (can use ``R=``, ``z=``)
- **Mix of spherical and cylindrical**: composite is cylindrical
- **Any component without a simple symmetry**: composite has no symmetry (must use Cartesian)

Examples
--------

A fully spherical system (bulge + halo)::

    >>> bulge = gp.HernquistPotential(m=1e10 * u.Msun, c=0.6 * u.kpc, units=galactic)
    >>> halo = gp.NFWPotential(m=1e12 * u.Msun, r_s=20 * u.kpc, units=galactic)
    >>> pot = gp.CompositePotential(bulge=bulge, halo=halo)
    >>> # Both spherical, composite is spherical
    >>> E = pot.energy(r=10 * u.kpc)
    >>> vc = pot.circular_velocity(r=np.linspace(1, 50, 100) * u.kpc)

A simple galaxy model (bulge + disk + halo)::

    >>> bulge = gp.HernquistPotential(m=2e10 * u.Msun, c=0.6 * u.kpc, units=galactic)
    >>> disk = gp.MiyamotoNagaiPotential(m=1e11 * u.Msun, a=3 * u.kpc, b=0.3 * u.kpc, units=galactic)
    >>> halo = gp.NFWPotential(m=1e12 * u.Msun, r_s=20 * u.kpc, units=galactic)
    >>> galaxy = gp.CompositePotential(bulge=bulge, disk=disk, halo=halo)
    >>> # Mix of spherical and cylindrical (disk), so composite is cylindrical
    >>> E = galaxy.energy(R=8 * u.kpc)  # Midplane energy
    >>> E = galaxy.energy(R=8 * u.kpc, z=0.5 * u.kpc)  # Off midplane

Computing a rotation curve for the galaxy model::

    >>> R = np.linspace(0.1, 20, 200) * u.kpc
    >>> v_circ = galaxy.circular_velocity(R=R)

Note that once you add a cylindrical component to spherical components, you can no
longer use ``r=`` (spherical) coordinates - you must use ``R=, z=`` (cylindrical)::

    >>> # This would raise an error:
    >>> galaxy.energy(r=10 * u.kpc)  # doctest: +SKIP
    ValueError: Invalid coordinate(s) for CylindricalSymmetry: {'r'}


Notes
=====

Return Coordinates
------------------

**Gradients and accelerations are always returned in Cartesian coordinates**,
even when using symmetry coordinate inputs. This design choice ensures:

For example::

    >>> grad = pot.gradient(r=10 * u.kpc)
    >>> assert grad.shape == (3, 1)
    >>> # Returns shape (3, 1) array: [dx, dy, dz] in Cartesian coords

Coordinate Validation
---------------------

Symmetry coordinates are validated to ensure they match the potential's symmetry. Trying
to use the wrong coordinates will raise an error::

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
