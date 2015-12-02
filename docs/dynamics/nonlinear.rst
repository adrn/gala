.. _nonlinear:

******************
Nonlinear dynamics
******************

Introduction
============

This module contains utilities for nonlinear dynamics. Currently, the only
implemented features enable you to compute estimates of the maximum
Lyapunov exponent for an orbit. In future releases, there will be features
for creating surface of sections and computing the full Lyapunov spectrum.

Some imports needed for the code below::

    import astropy.units as u
    import numpy as np
    import gary.potential as gp
    import gary.dynamics as gd
    from gary.units import galactic

Computing Lyapunov exponents
============================

There are two ways to compute Lyapunov exponents implemented in `gary.dynamics`.
In most cases, you'll want to use the `~gary.dynamics.fast_lyapunov_max` function
because the integration is implemented in C and is quite fast. This function only
works if the potential you are working with has functions implemented in C (e.g.,
it is a `~gary.potential.CPotentialBase` subclass). With a potential object and
a set of initial conditions::

    >>> pot = gp.
    >>> r =
    >>> w0 = gd.CartesianPhaseSpacePosition(pos=x*u.kpc,
                                            vel=v*u.km/u.s)

API
---
.. automodapi:: gary.dynamics.nonlinear
    :no-heading:
    :headings: ^^
