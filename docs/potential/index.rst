.. _potential:

*************************************************
Gravitational potentials (`streamteam.potential`)
*************************************************

Introduction
============

This subpackage provides a number of classes for working with parametric
gravitational potentials. There are `base classes`_ for defining custom
potentials, but more useful are the `built-in potentials`_. These are commonly
used potentials that have the potential form, gradient (and acceleration), and
(in some cases) Hessians available as methods. These are particularly useful
in combination with the :mod:`streamteam.integrate` subpackage.

Getting started with the built-in classes
=========================================

The built-in potentials are all initialized by passing in parameter values. To
see what parameters are available for a given potential, check the
documentation for the individual classes below. You must also specify a unit
system when initializing a potential. A unit system is a set of non-reducible
units and must at least contain Astropy Unit objects for length, mass, and
time, and optionally an angle unit. For example,
`usys = (u.kpc, u.Msun, u.Myr, u.radian)`.

All of the built-in potential objects have defined methods that evaluate
the value of the potential or the gradient/acceleration at a given
position(s)::

    >>> ptmass = PointMassPotential(m=1., x0=[0.,0.], usys=(u.Msun, u.au, u.yr))
    >>> ptmass
    <PointMassPotential: x0=[0.0, 0.0], m=1.00>
    >>> ptmass.value([1.,-1.])
    -27.922166224010091
    >>> ptmass.value([[1.,-1.],[2.,3.]])
    array([-27.92216622, -10.95197465])
    >>> ptmass.gradient([1.,-1.])
    array([ 13.96108311, -13.96108311])
    >>> ptmass.acceleration([1.,-1.])
    array([ -13.96108311, 13.96108311])

The position(s) must be specified in the same length units as specified in
the unit system.

The classes also allow plotting isopotential contours, either as 1D slices
or contour plots over 2D grids. To plot a 1D slice over the dimension of
interest, you pass in a grid of values to compute the potential on and
then just numerical values for the other dimensions. For example, to
make a 1D plot of the potential contour at :math:`y=0,z=1`::

    >>> p = sp.MiyamotoNagaiPotential(1E11, 6.5, 0.27, usys=(u.kpc, u.Msun, u.Myr))
    >>> fig,axes = p.plot_contours(grid=(np.linspace(-15,15,100), 0., 1.))

Produces a plot like:

.. image:: ../_static/potential/miyamoto-nagai-1d.png

To instead make a 2D contour plot along, for example, :math:`y=0`, pass in
a grid of values for :math:`x` and a grid of values for :math:`z` (the
meshgridding is taken care of internally)::

   >>> x = np.linspace(-15,15,100)
   >>> z = np.linspace(-5,5,100)
   >>> p.plot_contours(grid=(x, 1., z))

Produces a plot like:

.. image:: ../_static/potential/miyamoto-nagai-2d.png

Example:
========================

Reference/API
=============

.. _base:

Base classes
------------

.. autosummary::
   :nosignatures:
   :toctree: _potential/

   streamteam.potential.core.Potential
   streamteam.potential.core.CartesianPotential
   streamteam.potential.core.CompositePotential

-------------------------------------------------------------

.. _builtin:

Built-in potentials
-------------------

.. autosummary::
   :nosignatures:
   :toctree: _potential/

   streamteam.potential.builtin.HernquistPotential
   streamteam.potential.builtin.IsochronePotential
   streamteam.potential.builtin.MiyamotoNagaiPotential
   streamteam.potential.builtin.NFWPotential
   streamteam.potential.builtin.PointMassPotential
   streamteam.potential.builtin.LogarithmicPotential