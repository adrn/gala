.. include:: ../references.txt

.. _potential:

*************************************************
Gravitational potentials (`gala.potential`)
*************************************************

Introduction
============

This subpackage provides a number of classes for working with parametric models
of gravitational potentials. There are base classes for creating custom
potential classes, but there are a number of built-in potentials implemented in
C and Cython for speed. These are commonly used potentials that have methods for
computing, for example, the potential energy, gradient, density, or mass
profiles. These are particularly useful in combination with the
`~gala.integrate` and `~gala.dynamics` subpackages.

Also defined in this subpackage are a set of reference frames which can be used
for numerical integration of orbits in non-static reference frames. See the
page on :ref:`reference-frames` for more information.

For code blocks below and any pages linked below, I assume the following imports
have already been excuted::

    >>> import astropy.units as u
    >>> import numpy as np
    >>> import gala.potential as gp
    >>> from gala.units import galactic, solarsystem, dimensionless

Getting started: built-in potential classes
===========================================

The built-in potentials are all initialized by passing in keyword argument
parameter values as :class:`~astropy.units.Quantity` objects or as numeric
values in a specified unit system. To see what parameters are available for a
given potential, check the documentation for the individual classes below. You
must also specify a `~gala.units.UnitSystem` when initializing a potential. A
unit system is a set of non-reducible units that define the length, mass, time,
and angle units. A few common unit systems are built in to the package (e.g.,
``galactic``, ``solarsystem``, ``dimensionless``).

All of the built-in potential objects have defined methods to evaluate the value
of the potential energy and the gradient/acceleration at a given position(s).
For example, here we will create a potential object for a 2D point mass located
at the origin with unit mass::

    >>> ptmass = gp.KeplerPotential(m=1.*u.Msun, units=solarsystem)
    >>> ptmass
    <KeplerPotential: m=1.00 (AU,yr,solMass,rad)>

If you pass in parameters with different units, they will be converted to the
specified unit system::

    >>> gp.KeplerPotential(m=1047.6115*u.Mjup, units=solarsystem)
    <KeplerPotential: m=1.00 (AU,yr,solMass,rad)>

If no units are specified for a parameter, it is assumed to already be
consistent with the `~gala.units.UnitSystem` passed in::

    >>> gp.KeplerPotential(m=1., units=solarsystem)
    <KeplerPotential: m=1.00 (AU,yr,solMass,rad)>

The potential classes work well with the :mod:`astropy.units` framework, but to
ignore units you can use the `~gala.units.DimensionlessUnitSystem` by
importing::

    >>> from gala.units import dimensionless
    >>> gp.KeplerPotential(m=1., units=dimensionless)
    <KeplerPotential: m=1.00 (dimensionless)>

With a potential object, we can evaluate the potential energy at some position::

    >>> ptmass.energy([1.,-1.,0.]*u.au)
    <Quantity [-27.92216622] AU2 / yr2>

These functions also accept both :class:`~astropy.units.Quantity` objects or
plain :class:`~numpy.ndarray`-like objects (in which case the position is
assumed to be in the unit system of the potential)::

    >>> ptmass.value([1.,-1.,0.])
    <Quantity [-27.92216622] AU2 / yr2>

This also works for multiple positions by passing in a 2D position (but see
:ref:`conventions` for a description of the interpretation of different axes)::

    >>> pos = np.array([[1.,-1.,0],
    ...                 [2.,3.,0]]).T
    >>> ptmass.value(pos*u.au)
    <Quantity [-27.92216622,-10.95197465] AU2 / yr2>

We may also compute the gradient or acceleration::

    >>> ptmass.gradient([1.,-1.,0]*u.au) # doctest: +FLOAT_CMP
    <Quantity [[ 13.96108311],
               [-13.96108311],
               [  0.        ]] AU / yr2>
    >>> ptmass.acceleration([1.,-1.,0]*u.au) # doctest: +FLOAT_CMP
    <Quantity [[-13.96108311],
               [ 13.96108311],
               [ -0.        ]] AU / yr2>

Some of the potential objects also have methods implemented for computing the
corresponding mass density and the Hessian of the potential (matrix of 2nd
derivatives) at given locations. For example::

    >>> pot = gp.HernquistPotential(m=1E9*u.Msun, c=1.*u.kpc, units=galactic)
    >>> pot.density([1.,-1.,0]*u.kpc) # doctest: +FLOAT_CMP
    <Quantity [ 7997938.82200887] solMass / kpc3>
    >>> pot.hessian([1.,-1.,0]*u.kpc) # doctest: +SKIP
    <Quantity [[[ -4.68318131e-05],
                [  5.92743432e-04],
                [  0.00000000e+00]],

               [[  5.92743432e-04],
                [ -4.68318131e-05],
                [  0.00000000e+00]],

               [[  0.00000000e+00],
                [  0.00000000e+00],
                [  5.45911619e-04]]] 1 / Myr2>

.. These objects also provide more specialized methods such as
.. :meth:`~gala.potential.Potential.plot_contours`, for plotting isopotential
.. contours in both 1D and 2D, and :meth:`~gala.potential.Potential.mass_enclosed`,
.. which estimates the mass enclosed within a specified spherical radius.

`~gala.potential.Potential.plot_contours` supports plotting
either 1D slices or 2D contour plots of isopotentials. To plot a 1D slice
over the dimension of interest, pass in a grid of values for that dimension
and numerical values for the others. For example, to make a 1D plot of the
potential value as a function of :math:`x` position at :math:`y=0, z=1`::

    >>> p = gp.MiyamotoNagaiPotential(m=1E11, a=6.5, b=0.27, units=galactic)
    >>> p.plot_contours(grid=(np.linspace(-15,15,100), 0., 1.)) # doctest: +SKIP

.. plot::
    :align: center

    import astropy.units as u
    import numpy as np
    import gala.potential as gp
    from gala.units import galactic, solarsystem

    pot = gp.MiyamotoNagaiPotential(m=1E11, a=6.5, b=0.27, units=galactic)
    fig = pot.plot_contours(grid=(np.linspace(-15,15,100), 0., 1.))

To instead make a 2D contour plot over :math:`x` and :math:`z` along with
:math:`y=0`, pass in a 1D grid of values for :math:`x` and a 1D grid of values
for :math:`z` (the meshgridding is taken care of internally)::

    >>> x = np.linspace(-15,15,100)
    >>> z = np.linspace(-5,5,100)
    >>> p.plot_contours(grid=(x, 1., z)) # doctest: +SKIP

.. plot::
    :align: center

    import astropy.units as u
    import numpy as np
    import gala.potential as gp
    from gala.units import galactic, solarsystem

    pot = gp.MiyamotoNagaiPotential(m=1E11, a=6.5, b=0.27, units=galactic)
    x = np.linspace(-15,15,100)
    z = np.linspace(-5,5,100)
    pot.plot_contours(grid=(x, 1., z))

:meth:`~gala.potential.PotentialBase.mass_enclosed` is a method that
numerically estimates the mass enclosed within a spherical shell defined
by the specified position. This numerically estimates
:math:`\frac{d \Phi}{d r}` along the vector pointing at the specified position
and estimates the enclosed mass simply as
:math:`M(<r)\approx\frac{r^2}{G} \frac{d \Phi}{d r}`. This function can
be used to compute, for example, a mass profile::

    >>> import matplotlib.pyplot as pl
    >>> pot = gp.NFWPotential(m=1E11*u.Msun, r_s=20.*u.kpc, units=galactic)
    >>> pos = np.zeros((3,100)) * u.kpc
    >>> pos[0] = np.logspace(np.log10(20./100.), np.log10(20*100.), pos.shape[1]) * u.kpc
    >>> m_profile = pot.mass_enclosed(pos)
    >>> pl.loglog(pos[0], m_profile, marker=None) # doctest: +SKIP

.. plot::
    :align: center

    import astropy.units as u
    import numpy as np
    import gala.potential as gp
    from gala.units import galactic, solarsystem
    import matplotlib.pyplot as pl

    pot = gp.NFWPotential(m=1E11*u.Msun, r_s=20.*u.kpc, units=galactic)
    pos = np.zeros((3,100)) * u.kpc
    pos[0] = np.logspace(np.log10(20./100.), np.log10(20*100.), pos.shape[1]) * u.kpc
    m_profile = pot.mass_enclosed(pos)
    pl.loglog(pos[0], m_profile, marker=None) # doctest: +SKIP

Potential objects can be `pickled <https://docs.python.org/2/library/pickle.html>`_
and can therefore be stored for later use. However, pickles are saved as binary
files. It may be useful to save to or load from text-based specifications of
Potential objects. This can be done with :func:`gala.potential.save` and
:func:`gala.potential.load`, or with the :meth:`~gala.potential.PotentialBase.save`
and method::

    >>> from gala.potential import load
    >>> pot = gp.NFWPotential(m=6E11*u.Msun, r_s=20.*u.kpc,
    ...                       units=galactic)
    >>> pot.save("potential.yml")
    >>> load("potential.yml")
    <NFWPotential: m=6.00e+11, r_s=20.00 (kpc,Myr,solMass,rad)>

Using gala.potential
====================
More details are provided in the linked pages below:

.. toctree::
   :maxdepth: 2

   define-new-potential
   compositepotential

API
===

.. automodapi:: gala.potential
