.. _integrate_potential_example:

===========================================================================
Generating a mock stellar stream and converting to Heliocentric coordinates
===========================================================================

We first need to import some relevant packages::

   >>> import astropy.coordinates as coord
   >>> import astropy.units as u
   >>> import numpy as np
   >>> import gala.coordinates as gc
   >>> import gala.dynamics as gd
   >>> import gala.potential as gp
   >>> from gala.units import galactic

In the examples below, we will work use the ``galactic``
`~gala.units.UnitSystem`: as I define it, this is: :math:`{\rm kpc}`,
:math:`{\rm Myr}`, :math:`{\rm M}_\odot`.

We first create a potential object to work with. For this example, we'll
use a two-component potential: a Miyamoto-Nagai disk with a spherical NFW
potential to represent a dark matter halo.

   >>> pot = gp.CCompositePotential()
   >>> pot['disk'] = gp.MiyamotoNagaiPotential(m=6E10*u.Msun,
   ...                                         a=3.5*u.kpc, b=280*u.pc,
   ...                                         units=galactic)
   >>> pot['halo'] = gp.NFWPotential(m=1E12, r_s=20*u.kpc, units=galactic)

We'll use the Palomar 5 globular cluster and stream as a motivation for this
example. For the position and velocity of the cluster, we'll use
:math:`(\alpha, \delta) = (229, −0.124)~{\rm deg}` ([odenkirchen02]_),
:math:`d = 22.9~{\rm kpc}` ([bovy16]_),
:math:`v_r = -58.7~{\rm km}~{\rm s}^{-1}` ([bovy16]_), and
:math:`(\mu_{\alpha,*}, \mu_\delta) = (-2.296,-2.257)~{\rm mas}~{\rm yr}^{-1}`
([fritz15]_)::

   >>> c = coord.SkyCoord(ra=229 * u.deg, dec=-0.124 * u.deg,
   ...                    distance=22.9 * u.kpc)
   >>> v = coord.SphericalDifferential(d_lon=-2.296/np.cos(c.dec) * u.mas/u.yr,
   ...                                 d_lat=-2.257 * u.mas/u.yr,
   ...                                 d_distance=-58.7 * u.km/u.s)

We'll first convert this position and velocity to Galactocentric coordinates::

   >>> c_gc = c.transform_to(coord.Galactocentric).cartesian
   >>> v_gc = gc.vhel_to_gal(c, v)
   >>> c_gc, v_gc
   (<CartesianRepresentation (x, y, z) in kpc
        ( 7.69726478,  0.22748727,  16.41135761)>,
    <CartesianDifferential (d_x, d_y, d_z) in km / s
        (-45.27067041, -124.04384674, -16.05904468)>)
   >>> pal5 = gd.PhaseSpacePosition(pos=c_gc, vel=v_gc)

We can now integrate an orbit in the previously defined gravitational potential
using Pal 5's position and velocity as initial conditions. We'll integrate the
orbit backwards (using a negative time-step) from its present-day position for 4
Gyr::

   >>> pal5_orbit = pot.integrate_orbit(pal5, dt=-0.5*u.Myr, n_steps=8000)
   >>> pal5_orbit.plot()
   >>> fig = pal5_orbit.plot()

.. plot::
   :align: center
   :context: close-figs

   import astropy.coordinates as coord
   import astropy.units as u
   import numpy as np
   import gala.coordinates as gc
   import gala.dynamics as gd
   import gala.potential as gp
   from gala.units import galactic

   pot = gp.CCompositePotential()
   pot['disk'] = gp.MiyamotoNagaiPotential(m=6E10*u.Msun,
                                           a=3.5*u.kpc, b=280*u.pc,
                                           units=galactic)
   pot['halo'] = gp.NFWPotential(m=1E12, r_s=20*u.kpc, units=galactic)

   c = coord.SkyCoord(ra=229 * u.deg, dec=-0.124 * u.deg,
                      distance=22.9 * u.kpc)
   v = coord.SphericalDifferential(d_lon=-2.296/np.cos(c.dec) * u.mas/u.yr,
                                   d_lat=-2.257 * u.mas/u.yr,
                                   d_distance=-58.7 * u.km/u.s)

   c_gc = c.transform_to(coord.Galactocentric).cartesian
   v_gc = gc.vhel_to_gal(c, v)

   pal5 = gd.PhaseSpacePosition(pos=c_gc, vel=v_gc)
   pal5_orbit = pot.integrate_orbit(pal5, dt=-0.5*u.Myr, n_steps=8000)
   fig = pal5_orbit.plot()

We can now generate a :ref:`mock stellar stream <mockstreams>` using the orbit
of the progenitor system (the Pal 5 cluster). We'll generate a stream using the
prescription presented in [fardal15]_::

   >>> from gala.dynamics.mockstream import fardal_stream
   >>> stream = fardal_stream(pot, pal5_orbit[::-1], prog_mass=1E5*u.Msun,
   ...                        release_every=4)
   >>> fig = stream.plot(marker='.', s=1, alpha=0.25)

.. plot::
   :align: center
   :context: close-figs

   from gala.dynamics.mockstream import fardal_stream
   stream = fardal_stream(pot, pal5_orbit[::-1], prog_mass=5E4*u.Msun,
                          release_every=4)
   fig = stream.plot(marker='.', s=1, alpha=0.25)

We now have the model stream particle positions and velocities in a
Galactocentric coordinate frame. To convert these to observable, Heliocentric
coordinates, we have to specify a desired coordinate frame. We'll convert to the
ICRS coordinate system and plot some of the Heliocentric kinematic quantities::

   >>> stream_c, stream_v = stream.to_coord_frame(coord.ICRS)

.. plot::
   :align: center
   :context: close-figs

   style = dict(marker='.', s=1, alpha=0.5)

   fig, axes = plt.subplots(1, 2, figsize=(10,5), sharex=True)

   axes[0].scatter(stream_c.ra.degree,
                   stream_c.dec.degree, **style)
   axes[0].set_xlim(250, 220)
   axes[0].set_ylim(-15, 15)

   axes[1].scatter(stream_c.ra.degree,
                   stream_v.d_distance.to(u.km/u.s), **style)
   axes[1].set_xlim(250, 220)
   axes[1].set_ylim(-100, 0)

   axes[0].set_xlabel(r'$\alpha\,[{\rm deg}]$')
   axes[1].set_xlabel(r'$\alpha\,[{\rm deg}]$')
   axes[0].set_ylabel(r'$\delta\,[{\rm deg}]$')
   axes[1].set_ylabel(r'$v_r\,[{\rm km}\,{\rm s}^{-1}]$')

   fig.tight_layout()

References
==========

.. [odenkirchen02] `Odenkirchen et al. (2002) <https://arxiv.org/abs/astro-ph/0206276>`_
.. [fritz15] `Fritz & Kallivayali (2015) <https://arxiv.org/abs/1508.06647>`_
.. [bovy16] `Bovy et al. (2016) <https://arxiv.org/abs/1609.01298>`_
.. [fardal15] `Fardal, Huang, Weinberg (2015) <http://arxiv.org/abs/1410.1861>`_
