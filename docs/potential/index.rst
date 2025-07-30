.. include:: ../references.txt

.. module:: gala.potential

*************************************************
Gravitational potentials (`gala.potential`)
*************************************************

Introduction
============

This subpackage provides a number of classes for working with parametric models
of gravitational potentials. There are a number of built-in potentials
implemented in C and Cython (for speed), and there are base classes that allow
for easy creation of `new custom potential classes <define-new-potential.html>`_
in pure Python or by writing custom C/Cython extensions. The ``Potential``
objects have convenience methods for computing common dynamical quantities, for
example: potential energy, spatial gradient, density, or mass profiles. These
are particularly useful in combination with the `~gala.integrate` and
`~gala.dynamics` subpackages.

Also defined in this subpackage are a set of reference frames which can be used
for numerical integration of orbits in non-static reference frames. See the page
on :ref:`hamiltonian-reference-frames` for more information. ``Potential``
objects can be combined with a reference frame and stored in a
`~gala.potential.hamiltonian.Hamiltonian` object that provides an easy interface
to numerical orbit integration.

For the examples below the following imports have already been executed::

    >>> import astropy.units as u
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import gala.potential as gp
    >>> from gala.units import galactic, solarsystem, dimensionless

Getting Started: Built-in Methods of Potential Classes
======================================================

Potential classes are initialized by passing parameter values as
:class:`~astropy.units.Quantity` objects or as numeric values with a
specified unit system. You must also specify a `~gala.units.UnitSystem`
— a set of non-reducible units defining length, mass, time, and angle units.
Common unit systems are built in (e.g., ``galactic``, ``solarsystem``,
``dimensionless``).

For example, to create a Kepler potential (point mass) with mass = 1 solar mass::

    >>> ptmass = gp.KeplerPotential(m=1.0 * u.Msun, units=solarsystem)
    >>> ptmass
    <KeplerPotential: m=1.00 (AU,yr,solMass,rad)>

Parameters with different units are automatically converted to the specified
unit system::

    >>> gp.KeplerPotential(m=1047.6115 * u.Mjup, units=solarsystem)
    <KeplerPotential: m=1.00 (AU,yr,solMass,rad)>

Parameters without units are assumed to be in the specified unit system::

    >>> gp.KeplerPotential(m=1.0, units=solarsystem)
    <KeplerPotential: m=1.00 (AU,yr,solMass,rad)>

To work without units, use `~gala.units.DimensionlessUnitSystem` or pass
``None``::

    >>> gp.KeplerPotential(m=1.0, units=None)
    <KeplerPotential: m=1.00 (dimensionless)>

All built-in potential objects have methods to evaluate the potential energy
and gradient/acceleration at given positions. For example, to evaluate the
potential energy at ``(x, y, z) = (1, -1, 0) AU``::

    >>> ptmass.energy([1.0, -1.0, 0.0] * u.au)
    <Quantity [-27.91440236] AU2 / yr2>

These functions accept both :class:`~astropy.units.Quantity` objects and
plain array-like objects (assumed to be in the potential's unit system)::

    >>> ptmass.energy([1.0, -1.0, 0.0])
    <Quantity [-27.91440236] AU2 / yr2>

For multiple positions, pass a 2D array where each column is a position::

    >>> pos = np.array([[1.0, -1.0, 0], [2.0, 3.0, 0]]).T
    >>> ptmass.energy(pos * u.au)
    <Quantity [-27.91440236, -10.94892941] AU2 / yr2>

We can also compute the gradient or acceleration::

    >>> ptmass.gradient([1.0, -1.0, 0] * u.au)  # doctest: +FLOAT_CMP
    <Quantity [[ 13.95720118],
               [-13.95720118],
               [  0.        ]] AU / yr2>
    >>> ptmass.acceleration([1.0, -1.0, 0] * u.au)  # doctest: +FLOAT_CMP
    <Quantity [[-13.95720118],
               [ 13.95720118],
               [ -0.        ]] AU / yr2>

Most of the potential objects also have methods implemented for computing the
corresponding mass density and the Hessian of the potential (the matrix of 2nd
derivatives) at given locations. For example, with the
:class:`~gala.potential.potential.HernquistPotential`, we can evaluate both the
mass density and Hessian at the position ``(x, y, z) = (1, -1, 0) kpc``::

    >>> pot = gp.HernquistPotential(m=1e9 * u.Msun, c=1.0 * u.kpc, units=galactic)
    >>> pot.density([1.0, -1.0, 0] * u.kpc)  # doctest: +FLOAT_CMP
    <Quantity [7997938.82200887] solMass / kpc3>
    >>> pot.hessian([1.0, -1.0, 0] * u.kpc)  # doctest: +SKIP
    <Quantity [[[ -4.68318131e-05],
                [  5.92743432e-04],
                [  0.00000000e+00]],

               [[  5.92743432e-04],
                [ -4.68318131e-05],
                [  0.00000000e+00]],

               [[  0.00000000e+00],
                [  0.00000000e+00],
                [  5.45911619e-04]]] 1 / Myr2>

Another useful method is
:meth:`~gala.potential.potential.PotentialBase.mass_enclosed`, which numerically
estimates the mass enclosed within a spherical shell defined by the specified
position. This numerically estimates :math:`\frac{d \Phi}{d r}` along the vector
pointing at the specified position and estimates the enclosed mass simply as
:math:`M(<r)\approx\frac{r^2}{G} \frac{d \Phi}{d r}`. This function can be used
to compute, for example, a mass profile::

    >>> pot = gp.NFWPotential(m=1e11 * u.Msun, r_s=20.0 * u.kpc, units=galactic)
    >>> pos = np.zeros((3, 100)) * u.kpc
    >>> pos[0] = (
    ...     np.logspace(np.log10(20.0 / 100.0), np.log10(20 * 100.0), pos.shape[1])
    ...     * u.kpc
    ... )
    >>> m_profile = pot.mass_enclosed(pos)
    >>> plt.loglog(pos[0], m_profile, marker="")  # doctest: +SKIP
    >>> plt.xlabel("$r$ [{}]".format(pos.unit.to_string(format="latex")))  # doctest: +SKIP
    >>> plt.ylabel("$M(<r)$ [{}]".format(m_profile.unit.to_string(format="latex")))  # doctest: +SKIP

.. plot::
    :align: center
    :context: close-figs
    :width: 60%

    import astropy.units as u
    import numpy as np
    import gala.potential as gp
    from gala.units import galactic, solarsystem
    import matplotlib.pyplot as plt

    pot = gp.NFWPotential(m=1e11 * u.Msun, r_s=20.0 * u.kpc, units=galactic)
    pos = np.zeros((3, 100)) * u.kpc
    pos[0] = (
        np.logspace(np.log10(20.0 / 100.0), np.log10(20 * 100.0), pos.shape[1])
        * u.kpc
    )
    m_profile = pot.mass_enclosed(pos)

    plt.figure()
    plt.loglog(pos[0], m_profile, marker="")  # doctest: +SKIP
    plt.xlabel("$r$ [{}]".format(pos.unit.to_string(format="latex")))
    plt.ylabel("$M(<r)$ [{}]".format(m_profile.unit.to_string(format="latex")))
    plt.tight_layout()

Plotting Equipotential and Isodensity contours
==============================================

Potential objects provide methods for visualizing isopotential
(:meth:`~gala.potential.potential.PotentialBase.plot_contours`) and isodensity
(:meth:`~gala.potential.potential.PotentialBase.plot_density_contours`) contours.
These methods create either 1D slices or 2D contour plots.

For a 1D slice, pass a grid for one dimension and fixed values for others.
For example, to plot potential vs. :math:`x` at :math:`y=0, z=1`::

    >>> p = gp.MiyamotoNagaiPotential(m=1e11, a=6.5, b=0.27, units=galactic)
    >>> fig, ax = plt.subplots()  # doctest: +SKIP
    >>> p.plot_contours(
    ...     grid=(np.linspace(-15, 15, 100), 0.0, 1.0), marker="", ax=ax
    ... )  # doctest: +SKIP

.. plot::
    :align: center
    :context: close-figs
    :width: 90%

    pot = gp.MiyamotoNagaiPotential(m=1e11, a=6.5, b=0.27, units=galactic)

    fig, ax = plt.subplots(1, 1)  # doctest: +SKIP
    pot.plot_contours(
        grid=(np.linspace(-15, 15, 100), 0.0, 1.0), marker="", ax=ax
    )  # doctest: +SKIP
    E_unit = pot.units["energy"] / pot.units["mass"]
    ax.set_xlabel("$x$ [{}]".format(pot.units["length"].to_string(format="latex")))  # doctest: +SKIP
    ax.set_ylabel("$\Phi(x,0,1)$ [{}]".format(E_unit.to_string(format="latex")))  # doctest: +SKIP
    fig.tight_layout()

For 2D contour plots, pass 1D grids for two dimensions and a fixed value
for the third. For example, to plot :math:`x`-:math:`z` contours at :math:`y=0`::

    >>> fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    >>> x = np.linspace(-15, 15, 100)
    >>> z = np.linspace(-5, 5, 100)
    >>> p.plot_contours(grid=(x, 1.0, z), ax=ax)  # doctest: +SKIP

.. plot::
    :align: center
    :context: close-figs
    :width: 60%

    x = np.linspace(-15, 15, 100)
    z = np.linspace(-5, 5, 100)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    pot.plot_contours(grid=(x, 1.0, z), ax=ax)
    ax.set_xlabel("$x$ [{}]".format(pot.units["length"].to_string(format="latex")))
    ax.set_ylabel("$z$ [{}]".format(pot.units["length"].to_string(format="latex")))
    fig.tight_layout()

Saving / loading potential objects
==================================

Potential objects can be saved to and loaded from YAML files using the
:meth:`~gala.potential.potential.PotentialBase.save` method and
:func:`~gala.potential.potential.load` function::

    >>> from gala.potential import load
    >>> pot = gp.NFWPotential(m=6e11 * u.Msun, r_s=20.0 * u.kpc, units=galactic)
    >>> pot.save("potential.yml")
    >>> load("potential.yml")
    <NFWPotential: m=6.00e+11, r_s=20.00, a=1.00, b=1.00, c=1.00 (kpc,Myr,solMass,rad)>

Exporting potentials as ``sympy`` expressions
=============================================

Most of the potential classes can be exported to a `sympy` expression that can
be used to manipulate or evaluate the form of the potential. To access this
functionality, the potential classes have a
`~gala.potential.potential.PotentialBase.to_sympy` classmethod (note: this
requires `sympy` to be installed):

.. doctest-requires:: sympy

    >>> expr, vars_, pars = gp.LogarithmicPotential.to_sympy()
    >>> str(expr)
    '0.5*v_c**2*log(r_h**2 + z**2/q3**2 + y**2/q2**2 + x**2/q1**2)'

This method also returns a dictionary containing the coordinate variables used
in the expression as ``sympy`` symbols, here defined as ``vars_``:

.. doctest-requires:: sympy

    >>> vars_
    {"x": x, "y": y, "z": z}

A second dictionary containing the potential parameters as `sympy` symbols is
also returned, here defined as ``pars``:

.. doctest-requires:: sympy

    >>> pars
    {"v_c": v_c, "r_h": r_h, "q1": q1, "q2": q2, "q3": q3, "phi": phi, "G": G}

The expressions and variables returned can be used to perform operations on the
potential expression. For example, to create a `sympy` expression for the
gradient of the potential:

.. doctest-requires:: sympy

    >>> import sympy as sy
    >>> grad = sy.derive_by_array(expr, list(vars_.values()))
    >>> grad[0]  # dPhi/dx
    1.0 * v_c**2 * x / (q1**2 * (r_h**2 + z**2 / q3**2 + y**2 / q2**2 + x**2 / q1**2))


Using gala.potential
====================
More details are provided in the linked pages below:

.. toctree::
   :maxdepth: 1

   define-new-potential
   compositepotential
   origin-rotation
   hamiltonian-reference-frames
   scf

API
===

.. automodapi:: gala.potential.potential

.. automodapi:: gala.potential.frame.builtin

.. automodapi:: gala.potential.hamiltonian
