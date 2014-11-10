.. _integrate_potential_example:

Integrating and plotting an orbit in an NFW potential
=====================================================

We first need to import some relevant packages::

   import numpy as np
   from streamteam.coordinates import spherical_to_cartesian
   import streamteam.integrate as si
   import streamteam.potential as sp
   from streamteam.units import galactic

The variable ``galactic`` is defined and included in this package as as
short-hand for what I refer to as a Galactic unit system: :math:`{\rm kpc}`,
:math:`{\rm Myr}`, :math:`{\rm M}_\odot`. It is simply a tuple of
:class:`astropy.units.Unit` objects that define this unit system.

We will now create a potential object to work with. For this example, we'll
use a spherical NFW potential, parametrized by a scale radius and the
circular velocity at the scale radius::

   v_c = (200*u.km/u.s).decompose(galactic).value
   potential = sp.SphericalNFWPotential(v_c=v_c, r_s=10., units=units)

The easiest way to integrate an orbit in this potential is to use the
:meth:`~streamteam.potential.Potential.integrate_orbit` method, which accepts
a single set of (or array of) initial conditions and a specification for the
time-stepping and performs the integration for you::

   initial_conditions = np.array([10., 0, 0, 0, v_c, 0])
   t,orbit = potential.integrate_orbit(initial_conditions, dt=0.5, nsteps=10000)

This method returns an array of times, ``t``, and the orbit, ``orbit``.
By default, this method uses Leapfrog integration to compute the orbit
(:class:`~streamteam.integrate.LeapfrogIntegrator`), but you can optionally specify
a different integrator class as a keyword argument::

   t,orbit = potential.integrate_orbit(initial_conditions, dt=0.5, nsteps=10000,
                                       Integrator=si.DOPRI853Integrator)

We can integrate many orbits in parallel by passing in a 2D array of initial
conditions. Here, as an example, we'll generate some random initial
conditions by sampling from a Gaussian around our initial orbit::

   norbits = 1000
   stddev = [0.1,0.1,0.1,0.01,0.01,0.01] # 100 pc spatial scale, ~10 km/s velocity scale
   initial_conditions = np.random.normal(initial_conditions, stddev, size=(norbits,6))
   t,orbits = potential.integrate_orbit(initial_conditions, dt=0.5, nsteps=10000)

We'll now plot the final positions of these orbits over isopotential contours.
We start by using the :meth:`~streamteam.potential.Potential.plot_contours`
method of the ``potential`` object to plot the potential contours. This function
returns a :class:`~matplotlib.figure.Figure` object



Now we need to define an integrator object to compute an orbit. We'll use the
Leapfrog integration scheme implemented in the `integrate` subpackage. The
integrator assumes that the acceleration function accepts time and position,
but the potential acceleration method only accepts a position, so we define
a temporary (lambda) function to get around this::

   import streamteam.integrate as si
   acc = lambda t,x: potential.acceleration(x)
   integrator = si.LeapfrogIntegrator(acc)

We'll now compute orbits for two different initial conditions::

   x0 = np.array([[11.,6.,19.],[31.,0.,-4.]])
   v0 = ([[50.,0.,0.],[120.,-120.,375.]]*u.km/u.s).decompose(units).value
   w0 = np.hstack((x0,v0))
   t,ws = integrator.run(w0, dt=1., nsteps=10000)

And finally, over plot the orbits on the potential contours::

   import matplotlib.pyplot as plt
   from matplotlib import cm
   x = np.linspace(-50,50,200)
   z = np.linspace(-50,50,200)
   fig,ax = potential.plot_contours(grid=(x,0.,z), cmap=cm.gray_r)
   ax.plot(ws[:,0,0], ws[:,0,2], marker=None, lw=2., alpha=0.6)
   ax.plot(ws[:,1,0], ws[:,1,2], marker=None, lw=2., alpha=0.6)
   fig.set_size_inches(8,8)

.. image:: _static/examples/nfw.png
