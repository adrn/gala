.. testsetup:: *
    import astropy.units as u
    import numpy as np
    import gary.potential as gp
    import gary.dynamics as gd
    from gary.units import galactic

.. _orbits-in-detail:

*****************************************************
Orbit and phase-space position objects in more detail
*****************************************************

Introduction
============

The `astropy.units` subpackage is excellent for working with numbers and
associated units, however, dynamical quantities often contain many
quantities with mixed units. An example is a position in phase-space, which
may contain some quantities with length units and some quantities with
velocity units. The `~gary.dynamics.PhaseSpacePosition` and
`~gary.dynamics.Orbit` subclasses are designed to work with these structures.

Phase-space positions
=====================

It is often useful to represent full phase-space positions quantities jointly.
For example, if you need to transform the velocities to a new coordinate
representation or frame, the positions often enter into the transformations.
The `~gary.dynamics.PhaseSpacePosition` subclasses provide an interface for
handling these numbers. At present, only the
`~gary.dynamics.CartesianPhaseSpacePosition` is fully implemented.

To create a `~gary.dynamics.CartesianPhaseSpacePosition` object, pass in a
cartesian position and velocity to the initializer::

    >>> gd.CartesianPhaseSpacePosition(pos=[4.,8.,15.]*u.kpc,
                                       vel=[-150.,50.,15.]*u.km/u.s)
    <CartesianPhaseSpacePosition (3, 1)>

Of course, this works with arrays of positions and velocities as well::

    >>> x = np.random.uniform(-10,10,size=(3,128))
    >>> v = np.random.uniform(-200,200,size=(3,128))
    >>> gd.CartesianPhaseSpacePosition(pos=x*u.kpc,
                                       vel=v*u.km/u.s)
    <CartesianPhaseSpacePosition (3, 128)>

TODO: what can we do with these objects?

.. _references:

.. automodapi:: gary.dynamics.orbit

.. automodapi:: gary.dynamics.core
