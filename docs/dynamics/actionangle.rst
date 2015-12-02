.. _actionangle:

************************************************
Transforming to actions, angles, and frequencies
************************************************

Introduction
============

Regular orbits permit a (local) transformation to a set of canonical coordinates
such that the momenta are independent, isolating integrals of motion (the actions,
:math:`\boldsymbol{J}`) and the conjugate coordinate variables (the angles,
:math:`\boldsymbol{\theta}`) linearly increase time. Action-angle coordinates are
useful for a number of applications because Hamilton's equations -- the equations
of motion -- are so simple:

.. math::

    H &= H(\boldsymbol{J})\\
    \dot{\boldsymbol{J}} &= -\frac{\partial H}{\partial \boldsymbol{\theta}} = 0\\
    \dot{\boldsymbol{\theta}} &= \frac{\partial H}{\partial \boldsymbol{J}} = \boldsymbol{\Omega}(\boldsymbol{J}) = {\rm constant}

Analytic transformations from phase-space to action-angle coordinates are only
known for a few simple cases where the gravitational potential is separable or
has many symmetries. However, astronomical systems can often be triaxial or
have complex radial profiles that are not captured by these simple systems.
Here we have implemented the method described in [sanders14]_  for computing
actions and angles for an arbitrary numerically integrated orbit. We demonstrate
this method below with three orbits:

* :ref:`loop-axisymmetric`
* :ref:`loop-triaxial`
* :ref:`chaos-triaxial`

For the examples below, we will use the `~gary.units.galactic` unit system and
assume the following imports have been executed::

    import astropy.coordinates as coord
    import astropy.units as u
    import matplotlib.pyplot as pl
    import numpy as np
    import gary.potential as gp
    import gary.dynamics as gd
    from gary.units import galactic

.. _loop-axisymmetric:

A loop orbit in an axisymmetric potential
=========================================

For an example of an axisymmetric potential, we use a flattened logarithmic
potential:

.. math::

    \Phi(x,y,z) = \frac{1}{2}v_{\rm c}^2\ln (x^2 + y^2 + (z/q)^2 + r_h^2)

with parameters

.. math::

    v_{\rm c} &= 150~{\rm km}~{\rm s}^{-1}\\
    q &= 0.85\\
    r_h &= 0

For the orbit, we use initial conditions

.. math::

    \boldsymbol{r} &= (8, 0, 0)~{\rm kpc}\\
    \boldsymbol{v} &= (75, 150, 50)~{\rm km}~{\rm s}^{-1}

In code, we create a potential and set up our initial conditions::

    >>> pot = gp.LogarithmicPotential(v_c=150*u.km/u.s, q1=1., q2=1., q3=0.85, r_h=0,
                                      units=galactic)
    >>> w0 = gd.CartesianPhaseSpacePosition(pos=[8,0,0.]*u.kpc,
                                            vel=[75, 150, 50.]*u.km/u.s)

We will now integrate the orbit and plot it in the meridional plane::

    >>> w = pot.integrate_orbit(w0, dt=0.5, nsteps=10000)
    >>> cyl_pos, cyl_vel = w.represent_as(coord.CylindricalRepresentation)
    >>> fig,ax = pl.subplots(1,1,figsize=(6,6))
    >>> ax.plot(cyl_pos.rho.to(u.kpc).value, cyl_pos.z.to(u.kpc).value,
    ...         marker=None, linestyle='-') # doctest: +SKIP
    >>> ax.set_xlabel("R [kpc]")
    >>> ax.set_ylabel("z [kpc]")

.. plot::
    :align: center

    import astropy.coordinates as coord
    import astropy.units as u
    import matplotlib.pyplot as pl
    import numpy as np
    import gary.potential as gp
    import gary.dynamics as gd
    from gary.units import galactic

    pot = gp.LogarithmicPotential(v_c=150*u.km/u.s, q1=1., q2=1., q3=0.85, r_h=0,
                                  units=galactic)
    w0 = gd.CartesianPhaseSpacePosition(pos=[8,0,0.]*u.kpc,
                                        vel=[75, 150, 50.]*u.km/u.s)

    w = pot.integrate_orbit(w0, dt=0.5, nsteps=10000)
    cyl_pos, cyl_vel = w.represent_as(coord.CylindricalRepresentation)
    fig,ax = pl.subplots(1,1,figsize=(6,6))
    ax.plot(cyl_pos.rho.to(u.kpc).value, cyl_pos.z.to(u.kpc).value,
            marker=None, linestyle='-')
    ax.set_xlabel("R [kpc]")
    ax.set_ylabel("z [kpc]")

# TODO: left off here

We'll now fit a toy potential to the orbit by minimizing the dispersion
in energy::

    >>>

The orbit is shown in the meridional plane in the figure below (black). In red,
we show the orbit from the same initial conditions in the best-fitting Isochrone
potential (the toy potential for loop orbits):

.. .. image:: ../_static/dynamics/orbit_Rz_axisymmetricloop.png

For the "true" orbit in the potential of interest, we first compute the actions,
angles, and frequencies using the full orbit (500000 timesteps). We then break
the orbit into 100 evenly spaced, overlapping sub-sections and compute the actions
and frequencies for each sub-section of the orbit. Below we plot the percent
deviation of the actions computed for each sub-section with relation to the
actions computed for the total orbit, and the same for the frequencies. For this
orbit, the deviations are all :math:`\ll` 1%.

.. .. image:: ../_static/dynamics/action_hist_axisymmetricloop.png

.. .. image:: ../_static/dynamics/freq_hist_axisymmetricloop.png

.. _loop-triaxial:

A loop orbit in a triaxial potential
====================================

For a triaxial potential, we again use a logarithmic potential:

.. math::

    \Phi(x,y,z) = \frac{1}{2}v_{\rm c}^2\ln ((x/q_1)^2 + (y/q_2)^2 + (z/q_3)^2)

with :math:`v_{\rm c}=0.15`, :math:`q_1=1.3`, :math:`q_2=1.`, and :math:`q_1=0.85`.

.. _triaxialloop:

Loop orbit
^^^^^^^^^^

We use the initial conditions:

.. math::

    \boldsymbol{r} &= (8, 0, 0)\\
    \boldsymbol{v} &= (0.05, 0.175, 0.05)

which produces the orbit shown below (black). Again in red, we show the orbit
integrated from the same initial conditions in the best-fitting Isochrone
potential (the toy potential for loop orbits):

.. .. image:: ../_static/dynamics/orbit_xyz_triaxialloop.png

We repeat the same procedure as above by first computing quantities for the full
orbit and then for overlapping sub-sections of the orbit. There is more variation
in the values of the computed actions, possibly because we are truncating the
Fourier series with too few modes, but the variations are only a few percent
relative to the actions and frequencies computed from the full orbit:

.. .. image:: ../_static/dynamics/action_hist_triaxialloop.png

.. .. image:: ../_static/dynamics/freq_hist_triaxialloop.png

.. _triaxialchaotic:

.. _chaos-triaxial:

A chaotic orbit in a triaxial potential
=======================================

We use the initial conditions:

.. math::

    \boldsymbol{r} &= (5.5, 5.5, 0)\\
    \boldsymbol{v} &= (-0.02, 0.02, 0.11)

which produces the orbit shown below (black). In red, we show the orbit
integrated from the same initial conditions in the best-fitting triaxial
harmonic oscillator potential (the toy potential for box orbits):

.. .. image:: ../_static/dynamics/orbit_xyz_triaxialchaotic.png

We repeat the same procedure as above by first computing quantities for the full
orbit and then for overlapping sub-sections of the orbit. For this orbit, there
is no real definition of actions because the orbit is irregular -- it diffuses
stochastically through action space and gets trapped in resonances along the way.
This is clear in the deviation plots below, showing that the values of the actions
and frequencies oscillate and vary on many timescales:

.. .. image:: ../_static/dynamics/action_hist_triaxialchaotic.png

.. .. image:: ../_static/dynamics/freq_hist_triaxialchaotic.png

.. _references:

References
==========

.. [binneytremaine] Binney & Tremaine (2008) `Galactic Dynamics <http://press.princeton.edu/titles/8697.html>`_
.. [sanders14] Sanders & Binney (2014) `Actions, angles and frequencies for numerically integrated orbits <http://arxiv.org/abs/1401.3600>`_
.. [mcgill90] McGill & Binney (1990) `Torus construction in general gravitational potentials <http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1990MNRAS.244..634M&amp;data_type=PDF_HIGH&amp;whole_paper=YES&amp;type=PRINTER&amp;filetype=.pdf>`_
