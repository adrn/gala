.. _integrate_potential_example:

Integrating and plotting an orbit in an NFW potential
=====================================================

We first need to import some relevant packages::

   >>> import astropy.units as u
   >>> import numpy as np
   >>> import gala.integrate as gi
   >>> import gala.dynamics as gd
   >>> import gala.potential as gp
   >>> from gala.units import galactic

In the examples below, we will work use the ``galactic`` `~gala.units.UnitSystem`:
as I define it, this is: :math:`{\rm kpc}`, :math:`{\rm Myr}`, :math:`{\rm M}_\odot`.

We first create a potential object to work with. For this example, we'll
use a spherical NFW potential, parametrized by a scale radius and the
circular velocity at the scale radius::

   >>> pot = gp.SphericalNFWPotential(v_c=200*u.km/u.s, r_s=10.*u.kpc, units=galactic)

As a demonstration, we're going to first integrate a single orbit in this potential.

The easiest way to do this is to use the `~gala.potential.PotentialBase.integrate_orbit`
method of the potential object, which accepts a set of initial conditions and
a specification for the time-stepping. We'll define the initial conditions as a
`~gala.dynamics.CartesianPhaseSpacePosition` object::

   >>> ics = gd.CartesianPhaseSpacePosition(pos=[10,0,0.]*u.kpc,
   ...                                      vel=[0,175,0]*u.km/u.s)
   >>> orbit = pot.integrate_orbit(ics, dt=2., n_steps=2000)

This method returns a `~gala.dynamics.CartesianOrbit` object that contains an
array of times and the (6D) position at each time-step. By default, this method
uses Leapfrog integration to compute the orbit
(:class:`~gala.integrate.LeapfrogIntegrator`), but you can optionally specify
a different (e.g., more precise) integrator class as a keyword argument::

   >>> orbit = pot.integrate_orbit(ics, dt=2., n_steps=2000,
   ...                             Integrator=gi.DOPRI853Integrator)

We can integrate many orbits in parallel by passing in a 2D array of initial
conditions. Here, as an example, we'll generate some random initial
conditions by sampling from a Gaussian around the initial orbit (with a
positional scale of 100 pc, and a velocity scale of 1 km/s)::

   >>> norbits = 128
   >>> new_pos = np.random.normal(ics.pos.to(u.pc).value, 100., size=(3,norbits))*u.pc
   >>> new_vel = np.random.normal(ics.vel.to(u.km/u.s).value, 1, size=(3,norbits))*u.km/u.s
   >>> new_ics = gd.CartesianPhaseSpacePosition(pos=new_pos, vel=new_vel)
   >>> orbits = pot.integrate_orbit(new_ics, dt=2., n_steps=2000)

We'll now plot the final positions of these orbits over isopotential contours.
We use the :meth:`~gala.potential.Potential.plot_contours` method of the potential
object to plot the potential contours. This function returns a
:class:`~matplotlib.figure.Figure` object, which we can then use to over-plot
the orbit points::

   >>> grid = np.linspace(-15,15,64)
   >>> fig = pot.plot_contours(grid=(grid,grid,0), cmap='Greys')
   >>> ax = fig.axes[0] # grab the first plot axes
   >>> ax.plot(orbits.pos[0,-1], orbits.pos[1,-1], # this is x and y, the last timestep
   ...         marker='.', linestyle='none', alpha=0.5, color='#cc0000') # doctest: +SKIP

.. plot::
   :align: center

   import astropy.units as u
   import numpy as np
   import gala.integrate as gi
   import gala.dynamics as gd
   import gala.potential as gp
   from gala.units import galactic

   np.random.seed(42)

   pot = gp.SphericalNFWPotential(v_c=200*u.km/u.s, r_s=10.*u.kpc, units=galactic)

   ics = gd.CartesianPhaseSpacePosition(pos=[10,0,0.]*u.kpc,
                                        vel=[0,175,0]*u.km/u.s)
   orbit = pot.integrate_orbit(ics, dt=2., n_steps=2000)

   norbits = 1024
   new_pos = np.random.normal(ics.pos.to(u.pc).value, 100., size=(3,norbits))*u.pc
   new_vel = np.random.normal(ics.vel.to(u.km/u.s).value, 1., size=(3,norbits))*u.km/u.s
   new_ics = gd.CartesianPhaseSpacePosition(pos=new_pos, vel=new_vel)
   orbits = pot.integrate_orbit(new_ics, dt=2., n_steps=2000)

   grid = np.linspace(-15,15,64)
   fig = pot.plot_contours(grid=(grid,grid,0), cmap='Greys')
   ax = fig.axes[0] # grab the first plot axes
   ax.plot(orbits.pos[0,-1], orbits.pos[1,-1], # this is x and y, the last timestep
           marker='.', linestyle='none', alpha=0.75, color='#cc0000')

