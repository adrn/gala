.. _gala-getting-started:

***************
Getting Started
***************

Welcome to the `gala` documentation!

.. TODO: in the paragraph below, switch the matplotlib link to :mod:`matplotlib`
.. when they add a top-level module definition

For practical reasons, this documentation generally assumes that you are
familiar with the Python programming language, including numerical and
computational libraries like :mod:`numpy`, :mod:`scipy`, and `matplotlib
<https://matplotlib.org/>`_. If you need a refresher on Python programming, we
recommend starting with the `official Python tutorial
<https://docs.python.org/3/tutorial/>`_, but many other good resources are
available on the internet, such as tutorials and lectures specifically designed
for `using Python for scientific applications <https://scipy-lectures.org/>`_.

On this introductory page, we will demonstrate a few common use cases for `gala`
and give an overview of the package functionality. For the examples
below, we will assume that the following imports have already been executed
because these packages will be generally required::

    >>> import astropy.units as u
    >>> import numpy as np


Computing your first orbit
==========================

One of the most common use cases for `gala` is to compute an orbit for a star
within a mass model for the Milky Way. To do this, we need to specify two
things: (1) the model of the Milky Way that we would like to use to represent
the mass distribution, and (2) the initial conditions of the star's orbit.

Mass models in `gala` are specified using Python classes that represent
standard gravitational potential models. For example, most of the standard,
parametrized gravitational potential models introduced in :cite:t:`Binney2008`
are available as classes in the :mod:`gala.potential` module. The standard Milky
Way model recommended for use in `gala` is the
`~gala.potential.potential.MilkyWayPotential`, which is a pre-defined,
multi-component mass model with parameters set to fiducial values that match the
rotation curve of the Galactic disk and the mass profile of the dark matter
halo. We can create an instance of this model with the fiducial parameters by
instantiating the `~gala.potential.potential.MilkyWayPotential` class without
any input::

    >>> import gala.potential as gp
    >>> mw = gp.MilkyWayPotential()
    >>> mw
    <CompositePotential disk,bulge,nucleus,halo>

This model, by default, contains four distinct potential components as listed in
the output above: disk, bulge, nucleus, and halo components. You can configure
any of the parameters of these components, or create your own "composite"
potential model using other potential models defined in :mod:`gala.potential`,
but for now we will use the fiducial model as we defined it, the variable
``mw``.

All of the :mod:`gala.potential` class instances have a set of standard methods
that enable fast calculations of computed or derived quantities. For example,
we could compute the potential energy or the acceleration at a Cartesian
position near the Sun::

    >>> xyz = [-8., 0, 0] * u.kpc
    >>> mw.energy(xyz)  # doctest: +FLOAT_CMP
    <Quantity [-0.16440296] kpc2 / Myr2>
    >>> mw.acceleration(xyz)  # doctest: +FLOAT_CMP
    <Quantity [[ 0.00702262],
               [-0.        ],
               [-0.        ]] kpc / Myr2>

The values that are returned by most methods in `gala` are provided as Astropy
`~astropy.units.Quantity` objects, which represent numerical data with
associated physical units. `~astropy.units.Quantity` objects can be
re-represented in any equivalent units, so, for example, we could display the
energy or acceleration in other units::

    >>> E = mw.energy(xyz)
    >>> E.to((u.km/u.s)**2)  # doctest: +FLOAT_CMP
    <Quantity [-157181.98979398] km2 / s2>
    >>> acc = mw.acceleration(xyz)
    >>> acc.to(u.km/u.s / u.Myr)  # doctest: +FLOAT_CMP
    <Quantity [[ 6.86666358],
               [-0.        ],
               [-0.        ]] km / (Myr s)>

Now that we have a potential model, if we want to compute an orbit, we need to
specify a set of initial conditions to initialize the numerical orbit
integration. In `gala`, initial conditions and other positions in phase-space
(locations in position and velocity space) are defined using the
`~gala.dynamics.PhaseSpacePosition` class. This class allows a number of
possible inputs, but one of the most common inputs are Cartesian position and
velocity vectors. As an example orbit, we will use a position and velocity that
is close to the Sun's Galactocentric position and velocity::

    >>> import gala.dynamics as gd
    >>> w0 = gd.PhaseSpacePosition(pos=[-8.1, 0, 0.02] * u.kpc,
    ...                            vel=[13, 245, 8.] * u.km/u.s)

By convention, I typically use the variable ``w`` to represent phase-space
positions, so here ``w0`` is meant to imply "initial conditions." Note that,
when passing in Cartesian position and velocity values, we typically have to
pass them in as `~astropy.units.Quantity` objects (i.e., with units). This is
required whenever the potential class you are using has a unit system, which you
can check by calling the `~gala.potential.potential.PotentialBase.units`
attribute of your potential object::

    >>> mw.units
    <UnitSystem (kpc,Myr,solMass,rad)>

Here, our Milky Way potential model has a unit system with dimensional units.
Note that we could have used any length unit for the position and any velocity
unit for the velocity, because `gala` handles the unit conversions internally.

Now with a potential model defined and a set of initial conditions, we are set
to compute an orbit! To do this, we use the numerical integration system defined
in `gala.integrate`, but do so using the convenience interface available on any
Potential object through the
`~gala.potential.potential.PotentialBase.integrate_orbit()` method::

    >>> orbit = mw.integrate_orbit(w0, dt=1*u.Myr, t1=0, t2=2*u.Gyr)

By default, this method uses Leapfrog integration , which is a fast, symplectic
integration scheme. The returned object is an instance of the
`~gala.dynamics.Orbit` class, which is similar to the
`~gala.dynamics.PhaseSpacePosition` but represents a collection of phase-space
positions at times::

    >>> orbit
    <Orbit cartesian, dim=3, shape=(2000,)>

`~gala.dynamics.Orbit` objects have many of their own useful methods for
performing common tasks, like plotting an orbit::

    >>> orbit.plot()  # doctest: +SKIP

.. plot::
    :align: center
    :context: close-figs

    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    import gala.dynamics as gd
    import gala.potential as gp

    mw = gp.MilkyWayPotential()
    w0 = gd.PhaseSpacePosition(pos=[-8.1, 0, 0.02] * u.kpc,
                               vel=[13, 245, 8.] * u.km/u.s)
    orbit = mw.integrate_orbit(w0, dt=1*u.Myr, t1=0, t2=2*u.Gyr)

    orbit.plot()


Extensible and easy to define new potentials
============================================

New potentials can be easily defined by subclassing the base potential class,
`~gala.potential.PotentialBase`. For faster orbit integration and computation,
you can also define potentials with functions that evaluate its derived
quantities in C by subclassing `~gala.potential.CPotentialBase`. For fast
creation of potentials for quick testing, you can also create a potential
class directly from an equation that expresses the potential:

    >>> SHOPotential = gp.from_equation("1/2*k*x**2", vars="x", pars="k",
    ...                                 name='HarmonicOscillator')

(note: this requires `sympy`).

Classes created this way can then be instantiated and used like any other
`~gala.potential.PotentialBase` subclass:

    >>> pot = SHOPotential(k=1.)
    >>> pot.energy([1.])
    <Quantity [0.5]>

Extremely fast orbit integration
================================

Most of the guts of the `built-in potential classes <potential>`_ are
implemented in C, enabling extremely fast orbit integration for single or
composite potentials:

    >>> pot = gp.IsochronePotential(m=1E10*u.Msun, b=15.*u.kpc, units=galactic)
    >>> w0 = gd.PhaseSpacePosition(pos=[7.,0,0]*u.kpc,
    ...                            vel=[0.,50.,0]*u.km/u.s)
    >>> import timeit
    >>> timeit.timeit(lambda: gp.Hamiltonian(pot).integrate_orbit(w0, dt=0.5, n_steps=10000), number=100) / 100. # doctest: +SKIP
    0.0028513244865462184

For a composite potential:

    >>> bulge = gp.IsochronePotential(m=2E10*u.Msun, b=0.5*u.kpc, units=galactic)
    >>> disk = gp.MiyamotoNagaiPotential(m=6E10*u.Msun, a=3*u.kpc, b=0.26*u.kpc, units=galactic)
    >>> pot = gp.CCompositePotential(bulge=bulge, disk=disk)
    >>> timeit.timeit(lambda: gp.Hamiltonian(pot).integrate_orbit(w0, dt=0.5, n_steps=10000), number=100) / 100. # doctest: +SKIP
    0.0031369362445548177

Precise integrators
===================

The default orbit integration routine uses `~gala.integrate.LeapfrogIntegrator`,
but the high-order Dormand-Prince 853 integration scheme is also implemented as
`~gala.integrate.DOPRI853Integrator`:

    >>> orbit = gp.Hamiltonian(pot).integrate_orbit(w0, dt=0.5, n_steps=10000,
    ...                             Integrator=gi.DOPRI853Integrator)

Easy visualization
==================

Numerically integrated orbits can be easily visualized using the
`~gala.dynamics.Orbit.plot()` method:

    >>> orbit.plot() # doctest: +SKIP

.. plot::
    :align: center

    import astropy.units as u
    import numpy as np
    import gala.integrate as gi
    import gala.dynamics as gd
    import gala.potential as gp
    from gala.units import galactic

    bulge = gp.IsochronePotential(m=2E10*u.Msun, b=0.5*u.kpc, units=galactic)
    disk = gp.MiyamotoNagaiPotential(m=6E10*u.Msun, a=3*u.kpc, b=0.26*u.kpc, units=galactic)
    pot = gp.CCompositePotential(bulge=bulge, disk=disk)

    w0 = gd.PhaseSpacePosition(pos=[7.,0,0]*u.kpc,
                               vel=[0.,50.,0]*u.km/u.s)

    orbit = gp.Hamiltonian(pot).integrate_orbit(w0, dt=0.5, n_steps=10000,
                                Integrator=gi.DOPRI853Integrator)

    fig = orbit.plot()

Astropy units support
=====================

All functions and classes have Astropy unit support built in: they accept and
return `~astropy.units.Quantity` objects wherever possible. In addition, this
package uses an experimental new `~gala.units.UnitSystem` class for storing
systems of units and default representations.


Bibliography
============

.. bibliography::
    :cited:
