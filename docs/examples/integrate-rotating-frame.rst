.. _integrate_rotating_frame:

==================================================
Integrating an orbit in a rotating reference frame
==================================================

We first need to import some relevant packages::

   >>> import astropy.units as u
   >>> import matplotlib.pyplot as plt
   >>> import numpy as np
   >>> import gala.integrate as gi
   >>> import gala.dynamics as gd
   >>> import gala.potential as gp
   >>> from gala.units import galactic
   >>> from scipy.optimize import minimize

-----------------------------------
Orbits in a barred galaxy potential
-----------------------------------

In the example below, we will work use the ``galactic``
`~gala.units.UnitSystem`: as I define it, this is: :math:`{\rm kpc}`,
:math:`{\rm Myr}`, :math:`{\rm M}_\odot`.

For this example, we'll use a two-component potential: a Miyamoto-Nagai disk
with a triaxial (weak) bar component ([longmurali]_). We'll set the mass of the
bar to be 1/6 the mass of the disk component. We'll set the long-axis
scale length of the bar to the co-rotation radius, and arbitrarily set the other
scale lengths to reasonable values. We therefore first need to specify a pattern
speed for the bar. We'll use :math:`\Omega_p = 40~{\rm km}~{\rm s}^{-1}~{\rm
kpc}^{-1}`. We then have to solve for the co-rotation radius using the circular
::

   >>> disk = gp.MiyamotoNagaiPotential(m=6E10*u.Msun,
   ...                                  a=3.5*u.kpc, b=280*u.pc,
   ...                                  units=galactic)
   >>> Omega = 40. * u.km/u.s/u.kpc
   >>> def corot_func(r, Omega, potential):
   ...     x = r[0]
   ...     xyz = [x, 0., 0.] * u.kpc
   ...     v_bar = Omega * (x * u.kpc)
   ...     v_circ = disk.circular_velocity(xyz).to(u.km/u.s)
   ...     return (v_bar - v_circ).value ** 2
   >>> res = minimize(corot_func, x0=4., args=(Omega, disk))
   >>> corot_rad = res.x[0] * u.kpc
   >>> corot_rad # doctest: +FLOAT_CMP
   <Quantity 3.9175069560019287 kpc>

We can now define the bar component::

   >>> bar = gp.LongMuraliBarPotential(m=1E10*u.Msun, a=corot_rad,
   ...                                 b=0.8*u.kpc, c=0.25*u.kpc,
   ...                                 units=galactic)
   >>> pot = gp.CCompositePotential()
   >>> pot['disk'] = disk
   >>> pot['bar'] = bar
   >>> grid = np.linspace(-10, 10, 128)
   >>> fig,ax = plt.subplots(1, 1, figsize=(5,5))
   >>> fig = pot.plot_contours(grid=(grid, grid, 0), ax=ax)

.. plot::
   :align: center
   :context: close-figs

   import astropy.units as u
   import matplotlib.pyplot as plt
   import numpy as np
   import gala.integrate as gi
   import gala.dynamics as gd
   import gala.potential as gp
   from gala.units import galactic
   from scipy.optimize import minimize

   disk = gp.MiyamotoNagaiPotential(m=6E10*u.Msun,
                                    a=3.5*u.kpc, b=280*u.pc,
                                    units=galactic)
   Omega = 40. * u.km/u.s/u.kpc
   def corot_func(r, Omega, potential):
       x = r[0]
       xyz = [x, 0., 0.] * u.kpc
       v_bar = Omega * (x * u.kpc)
       v_circ = disk.circular_velocity(xyz).to(u.km/u.s)
       return (v_bar - v_circ).value ** 2
   res = minimize(corot_func, x0=4., args=(Omega, disk))
   corot_rad = res.x[0] * u.kpc

   bar = gp.LongMuraliBarPotential(m=1E10*u.Msun, a=corot_rad,
                                   b=0.8*u.kpc, c=0.25*u.kpc,
                                   units=galactic)
   pot = gp.CCompositePotential()
   pot['disk'] = disk
   pot['bar'] = bar
   grid = np.linspace(-10, 10, 128)
   fig,ax = plt.subplots(1, 1, figsize=(5,5))
   fig = pot.plot_contours(grid=(grid, grid, 0), ax=ax)

----------------------------------------------------
Orbits in the circular restricted three-body problem
----------------------------------------------------

TODO:

References
==========

.. [longmurali] `Long & Murali (1992) <http://adsabs.harvard.edu/abs/1992ApJ...397...44L>`_
