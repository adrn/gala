.. _orbits-in-detail:

*****************************************************
Orbit and phase-space position objects in more detail
*****************************************************

We'll assume the following imports have already been executed::

    >>> import astropy.units as u
    >>> import numpy as np
    >>> import gala.potential as gp
    >>> import gala.dynamics as gd
    >>> from gala.dynamics.extern import (CylindricalRepresentation,
    ...                                   CylindricalDifferential)
    >>> from gala.units import galactic
    >>> np.random.seed(42)

Introduction
============

The `astropy.units` subpackage is excellent for working with numbers and
associated units, but dynamical quantities often contain many
quantities with mixed units. An example is a position in phase-space, which
may contain some quantities with length units and some quantities with
velocity or momentum units. The `~gala.dynamics.PhaseSpacePosition` and
`~gala.dynamics.Orbit` classes are designed to work with these data structures
and provide a consistent API for visualizing and computing further dynamical
quantities. Click these shortcuts to jump to a section below, or start reading
below:

 * :ref:`phase-space-position`
 * :ref:`orbit`

.. _phase-space-position:

Phase-space positions
=====================

The `~gala.dynamics.PhaseSpacePosition` class provides an interface for
representing full phase-space positions--coordinate positions and momenta
(velocities). This class is useful for encapsulating initial condition
information and for transforming phase-space positions to new coordinate
representations or frames.

The easiest way to create a `~gala.dynamics.PhaseSpacePosition` object is to
pass in a pair of `~astropy.units.Quantity` objects that represent the 3-space
position and velocity vectors::

    >>> gd.PhaseSpacePosition(pos=[4.,8.,15.]*u.kpc,
    ...                       vel=[-150.,50.,15.]*u.km/u.s)
    <PhaseSpacePosition, shape=(), frame=None>

By default, these are interpreted as Cartesian coordinates and velocities. This
works with arrays of positions and velocities as well::

    >>> x = np.random.uniform(-10, 10, size=(3,8))
    >>> v = np.random.uniform(-200, 200, size=(3,8))
    >>> w = gd.PhaseSpacePosition(pos=x * u.kpc,
    ...                           vel=v * u.km/u.s)
    >>> w
    <PhaseSpacePosition, shape=(8,), frame=None>

This is interpreted as 8, 6-dimensional phase-space positions.

The class internally stores the positions and velocities as
`~astropy.coordinates.BaseRepresentation` and
`~astropy.coordinates.BaseDifferential` subclasses. All of the components of
these classes are added as attributes of the phase-space position class for
convenience. For example, to access the ``x`` component of the position and
the ``d_x`` component of the velocity::

    >>> w.x, w.d_x # doctest: +FLOAT_CMP
    (<Quantity [-2.50919762, 9.01428613, 4.63987884, 1.97316968,-6.87962719,
                -6.88010959,-8.83832776, 7.32352292] kpc>,
     <Quantity [ -17.57200631, 114.07038456,-120.13048714,   5.69377537,
                  36.96582754,-181.41983491,  43.01794076,-131.79035053] km / s>)

The default representation is Cartesian, but the class can also be instantiated
with representation objects::

    >>> pos = CylindricalRepresentation(rho=np.random.uniform(1., 10., size=8) * u.kpc,
    ...                                 phi=np.random.uniform(0, 2*np.pi, size=8) * u.rad,
    ...                                 z=np.random.uniform(-1, 1., size=8) * u.kpc)
    >>> vel = CylindricalDifferential(d_rho=np.random.uniform(100, 150., size=8) * u.km/u.s,
    ...                               d_phi=np.random.uniform(-0.1, 0.1, size=8) * u.rad/u.Myr,
    ...                               d_z=np.random.uniform(-15, 15., size=8) * u.km/u.s)
    >>> w = gd.PhaseSpacePosition(pos=pos, vel=vel)
    >>> w
    <PhaseSpacePosition, shape=(8,), frame=None>
    >>> w.rho # doctest: +SKIP
    <Quantity [ 4.01701598, 3.59795742, 4.77271308, 9.93452517, 6.64624698,
                5.52105132, 9.44344862, 5.91498428] kpc>

We can easily transform the full phase-space vector to new representations or
coordinate frames. These transformations use the :mod:`astropy.coordinates`
`representations framework <http://docs.astropy.org/en/latest/coordinates/skycoord.html#astropy-skycoord-representations>`_
and the velocity transforms from `gala.coordinates`.

    >>> from astropy.coordinates import CartesianRepresentation
    >>> cart = w.represent_as(CartesianRepresentation)
    >>> cart.x # doctest: +SKIP
    <Quantity [ 3.26170991, 2.99882353,-4.77123661, 9.74250792,-6.16304849,
                1.24929222,-9.23722316,-4.07817788] kpc>

[TODO: frame transforming]

There is also support for transforming the cartesian positions and velocities
(assumed to be in a `~astropy.coordinates.Galactocentric` frame) to any of
the other coordinate frames. The transformation returns two objects: an
initialized coordinate frame for the position, and a tuple of
:class:`~astropy.units.Quantity` objects for the velocity. Here, velocities
are represented in angular velocities for the velocities conjugate to angle
variables. For example, in the below transformation to
:class:`~astropy.coordinates.Galactic` coordinates, the returned velocity
object is a tuple with proper motions and radial velocity,
:math:`(\mu_l, \mu_b, v_r)`::

    >>> from astropy.coordinates import Galactic
    >>> gal_c, gal_v = w.to_coord_frame(Galactic)
    >>> gal_c # doctest: +SKIP
    <Galactic Coordinate: (l, b, distance) in (deg, deg, kpc)
        [(17.67673481, 31.15412806, 10.19473889),
    ...etc.
    >>> gal_v[0].unit, gal_v[1].unit, gal_v[2].unit
    (Unit("mas / yr"), Unit("mas / yr"), Unit("km / s"))

We can easily plot projections of the positions using the
`~gala.dynamics.CartesianPhaseSpacePosition.plot` method::

    >>> fig = w.plot()

.. plot::
    :align: center

    import astropy.units as u
    import numpy as np
    import gala.dynamics as gd
    np.random.seed(42)
    x = np.random.uniform(-10,10,size=(3,128))
    v = np.random.uniform(-200,200,size=(3,128))
    w = gd.CartesianPhaseSpacePosition(pos=x*u.kpc,
                                       vel=v*u.km/u.s)
    fig = w.plot()

This is a thin wrapper around the `~gala.dynamics.three_panel`
function and any keyword arguments are passed through to that function::

    >>> fig = w.plot(marker='o', s=40, alpha=0.5)

.. plot::
    :align: center

    import astropy.units as u
    import numpy as np
    import gala.dynamics as gd
    np.random.seed(42)
    x = np.random.uniform(-10,10,size=(3,128))
    v = np.random.uniform(-200,200,size=(3,128))
    w = gd.CartesianPhaseSpacePosition(pos=x*u.kpc,
                                       vel=v*u.km/u.s)
    fig = w.plot(marker='o', s=40, alpha=0.5)

Phase-space position API
------------------------
.. automodapi:: gala.dynamics.core
    :no-heading:
    :headings: ^^

.. _orbit:

Orbits
======

The `~gala.dynamics.Orbit` subclasses all inherit the functionality described
above from `~gala.dynamics.PhaseSpacePosition`, but similarly, at present only the
`~gala.dynamics.CartesianOrbit` is fully implemented. There are some differences
between the methods and some functionality that is particular to the orbit classes.

A `~gala.dynamics.CartesianOrbit` is initialized much like the
`~gala.dynamics.CartesianPhaseSpacePosition`. `~gala.dynamics.CartesianOrbit`s can be
created with just position and velocity information, however now the
interpretation of the input object shapes is different. Whereas an input position with
shape ``(2,128)`` to a `~gala.dynamics.CartesianPhaseSpacePosition` represents
128, 2D positions, for an orbit it would represent a single orbit's positions
at 128 timesteps::

    >>> t = np.linspace(0,10,128)
    >>> pos,vel = np.zeros((2,128)),np.zeros((2,128))
    >>> pos[0] = np.cos(t)
    >>> pos[1] = np.sin(t)
    >>> vel[0] = -np.sin(t)
    >>> vel[1] = np.cos(t)
    >>> orbit = gd.CartesianOrbit(pos=pos*u.kpc, vel=vel*u.km/u.s)
    >>> orbit
    <CartesianOrbit N=2, shape=(128,)>

To create a single object that contains multiple orbits, the input position object
should have 3 axes. The last axis (``axis=2``) contains each orbit. So, an input
position with shape ``(2,128,16)`` would represent 16, 2D orbits with 128 timesteps::

    >>> t = np.linspace(0,10,128)
    >>> pos,vel = np.zeros((2,128,16)),np.zeros((2,128,16))
    >>> Omega = np.random.uniform(size=16)
    >>> pos[0] = np.cos(Omega[np.newaxis]*t[:,np.newaxis])
    >>> pos[1] = np.sin(Omega[np.newaxis]*t[:,np.newaxis])
    >>> vel[0] = -np.sin(Omega[np.newaxis]*t[:,np.newaxis])
    >>> vel[1] = np.cos(Omega[np.newaxis]*t[:,np.newaxis])
    >>> orbit = gd.CartesianOrbit(pos=pos*u.kpc, vel=vel*u.km/u.s)
    >>> orbit
    <CartesianOrbit N=2, shape=(128, 16)>

To make full use of the orbit functionality, you must also pass in an array with
the time values and an instance of a `~gala.potential.PotentialBase` subclass that
represents the potential that the orbit was integrated in::

    >>> pot = gp.PlummerPotential(m=1E10, b=1., units=galactic)
    >>> orbit = gd.CartesianOrbit(pos=pos*u.kpc, vel=vel*u.km/u.s,
    ...                           t=t*u.Myr, potential=pot)

(note, in this case ``pos`` and ``vel`` were not generated from integrating
an orbit in the potential ``pot``!) Orbit objects
are returned by the `~gala.potential.PotentialBase.integrate_orbit` method
of potential objects that already have the ``time`` and ``potential`` set::

    >>> pot = gp.PlummerPotential(m=1E10, b=1., units=galactic)
    >>> w0 = gd.CartesianPhaseSpacePosition(pos=[10.,0,0]*u.kpc,
    ...                                     vel=[0.,75,0]*u.km/u.s)
    >>> orbit = pot.integrate_orbit(w0, dt=1., n_steps=500)
    >>> orbit
    <CartesianOrbit N=3, shape=(501,)>
    >>> orbit.t # doctest: +SKIP
    <Quantity [   0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,
                 10.,  11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,
    ...etc.
    >>> orbit.potential
    <PlummerPotential: m=1.00e+10, b=1.00 (kpc,Myr,solMass,rad)>

From an Orbit object, we can quickly compute quantities like the angular momentum,
and estimates for the pericenter, apocenter, eccentricity of the orbit. Estimates
for the latter few get better with smaller timesteps::

    >>> orbit = pot.integrate_orbit(w0, dt=0.1, n_steps=100000)
    >>> np.mean(orbit.angular_momentum(), axis=1) # doctest: +FLOAT_CMP
    <Quantity [ 0.        , 0.        , 0.76703412] kpc2 / Myr>
    >>> orbit.eccentricity() # doctest: +FLOAT_CMP
    <Quantity 0.3191563009914265>
    >>> orbit.pericenter() # doctest: +FLOAT_CMP
    <Quantity 10.00000005952518 kpc>
    >>> orbit.apocenter() # doctest: +FLOAT_CMP
    <Quantity 19.375317870528118 kpc>

Just like above, we can quickly visualize an orbit using the
`~gala.dynamics.CartesianOrbit.plot` method::

    >>> fig = orbit.plot()

.. plot::
    :align: center

    import astropy.units as u
    import gala.dynamics as gd
    import gala.potential as gp
    from gala.units import galactic

    pot = gp.PlummerPotential(m=1E10, b=1., units=galactic)
    w0 = gd.CartesianPhaseSpacePosition(pos=[1.,0,0]*u.kpc,
                                        vel=[0.,50,0]*u.km/u.s)
    orbit = pot.integrate_orbit(w0, dt=1., n_steps=500)
    fig = orbit.plot()

This is a thin wrapper around the `~gala.dynamics.plot_projections`
function and any keyword arguments are passed through to that function::

    >>> fig = orbit.plot(linewidth=4., alpha=0.5, color='r')

.. plot::
    :align: center

    import astropy.units as u
    import gala.dynamics as gd
    import gala.potential as gp
    from gala.units import galactic

    pot = gp.PlummerPotential(m=1E10, b=1., units=galactic)
    w0 = gd.CartesianPhaseSpacePosition(pos=[1.,0,0]*u.kpc,
                                        vel=[0.,50,0]*u.km/u.s)
    orbit = pot.integrate_orbit(w0, dt=1., n_steps=500)
    fig = orbit.plot(linewidth=4., alpha=0.5, color='r')
    fig.axes[0].set_xlim(-1.5,1.5)
    fig.axes[0].set_ylim(-1.5,1.5)


Orbit API
---------
.. automodapi:: gala.dynamics.orbit
    :no-heading:
    :headings: ^^
