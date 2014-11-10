.. _integrate_potential_example:

Example: Integrating and plotting an orbit in an NFW potential
==============================================================

First we will define the potential object. We'll just use a spherical NFW
halo by setting the axis ratios to unity::

   import astropy.units as u
   import numpy as np
   import streamteam.potential as sp

   units = (u.kpc, u.Msun, u.Myr)
   v_h = (250*u.km/u.s).decompose(units).value
   potential = sp.SphericalNFWPotential(v_h=v_h, r_h=10., units=units)

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
