.. include:: ../references.txt

.. _potential:

*************************************************
Gravitational potentials (`gary.potential`)
*************************************************

Introduction
============

This subpackage provides a number of classes for working with parametric
gravitational potentials. There are `base classes`_ for defining custom
potentials (see :ref:`custompotential` for more information), but more
useful are the `built-in potentials`_. These are commonly used potentials
that have methods for computing the potential value, gradient, and (in some
cases) Hessian. These are particularly useful in combination with
the `~gary.integrate` subpackage.

For code blocks below and any pages linked below, I assume the following
imports have already been excuted::

    >>> import astropy.units as u
    >>> import numpy as np
    >>> import gary.potential as gp
    >>> from gary.units import galactic, solarsystem

Getting started: built-in potential classes
===========================================

The built-in potentials are all initialized by passing in keyword argument
parameter values. To see what parameters are available for a given potential,
check the documentation for the individual classes below. You must also specify
a `~gary.units.UnitSystem` when initializing a potential. A unit system is a set of
non-reducible units that define the length, mass, time, and angle units. A few
common unit systems are built in to the package (e.g., ``solarsystem``).

All of the built-in potential objects have defined methods that evaluate
the value of the potential and the gradient/acceleration at a given
position(s). For example, here we will create a potential object for a
2D point mass located at the origin with unit mass::

    >>> ptmass = gp.PointMassPotential(m=1., x0=[0.,0.],
    ...                                units=solarsystem)
    >>> ptmass
    <PointMassPotential: x0=[0.0, 0.0], m=1.00>

We can then evaluate the value of the potential at some other position (note: the
position array is assumed to be in the unit system of the potential)::

    >>> ptmass.value([1.,-1.])
    array([-27.92216622])

Or at multiple positions, by passing in a 2D array::

    >>> pos = np.array([[1.,-1.],
    ...                 [2.,3.],
    ...                 [12.,-2.]]).T
    >>> ptmass.value(pos)
    array([-27.92216622, -10.95197465,  -3.24588589])

We may also compute the gradient of the potential or acceleration due to the potential::

    >>> ptmass.gradient([1.,-1.])
    array([[ 13.96108311, -13.96108311]])
    >>> ptmass.acceleration([1.,-1.])
    array([[-13.96108311,  13.96108311]])

The position(s) must be specified in the same length units as specified in
the unit system.

.. These objects also provide more specialized methods such as
.. :meth:`~gary.potential.Potential.plot_contours`, for plotting isopotential
.. contours in both 1D and 2D, and :meth:`~gary.potential.Potential.mass_enclosed`,
.. which estimates the mass enclosed within a specified spherical radius.

`~gary.potential.Potential.plot_contours` supports plotting
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
    import gary.potential as gp
    from gary.units import galactic, solarsystem

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
    import gary.potential as gp
    from gary.units import galactic, solarsystem

    pot = gp.MiyamotoNagaiPotential(m=1E11, a=6.5, b=0.27, units=galactic)
    x = np.linspace(-15,15,100)
    z = np.linspace(-5,5,100)
    pot.plot_contours(grid=(x, 1., z))

:meth:`~gary.potential.PotentialBase.mass_enclosed` is a method that
numerically estimates the mass enclosed within a spherical shell defined
by the specified position. This numerically estimates
:math:`\frac{d \Phi}{d r}` along the vector pointing at the specified position
and estimates the enclosed mass simply as
:math:`M(<r)\approx\frac{r^2}{G} \frac{d \Phi}{d r}`. This function can
be used to compute, for example, a mass profile::

    >>> import matplotlib.pyplot as pl
    >>> pot = gp.SphericalNFWPotential(v_c=0.5, r_s=20., units=galactic)
    >>> pos = np.zeros((3,100))
    >>> pos[0] = np.logspace(np.log10(20./100.), np.log10(20*100.), pos.shape[1])
    >>> m_profile = pot.mass_enclosed(pos)
    >>> pl.loglog(pos, m_profile, marker=None) # doctest: +SKIP

.. plot::
    :align: center

    import astropy.units as u
    import numpy as np
    import gary.potential as gp
    from gary.units import galactic, solarsystem
    import matplotlib.pyplot as pl

    pot = gp.SphericalNFWPotential(v_c=0.5, r_s=20., units=galactic)
    pos = np.zeros((3,100))
    pos[0] = np.logspace(np.log10(20./100.), np.log10(20*100.), pos.shape[1])
    m_profile = pot.mass_enclosed(pos)
    pl.loglog(pos[0], m_profile, marker=None) # doctest: +SKIP

Potential objects can be `pickled <https://docs.python.org/2/library/pickle.html>`_
and can therefore be stored for later use. However, pickles are saved as binary
files. It may be useful to save to or load from text-based specifications of
Potential objects. This can be done with :func:`gary.potential.save` and
:func:`gary.potential.load`, or with the :meth:`~gary.potential.PotentialBase.save`
and method::

    >>> from gary.potential import load
    >>> pot = SphericalNFWPotential(v_c=0.5, r_s=20.,
    ...                             units=(u.kpc, u.Msun, u.Myr))
    >>> pot.save("potential.yml")
    >>> load("potential.yml")
    <SphericalNFWPotential: r_s=20.00, v_c=0.50>

Further information
===================
More detailed information on using the package is provided on separate pages,
listed below.

.. toctree::
   :maxdepth: 2

   compositepotential
   custompotential

API
===

.. automodapi:: gary.potential
