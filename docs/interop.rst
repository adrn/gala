.. _gala-interop:

*********************************************
Interoperability with Other Dynamics Packages
*********************************************

Gala provides interfaces with other common Galactic dynamics packages, which
enables easily converting objects between these packages. Some examples are
shown below. As always, if something does not work as expected or you would like
more interoperability with any of these packages, please `open an issue
<https://github.com/adrn/gala/issues/new>`_ on GitHub.

Here are some imports we will use below in examples::

    >>> import astropy.units as u
    >>> import numpy as np
    >>> import gala.dynamics as gd
    >>> import gala.potential as gp
    >>> from gala.units import galactic

Galpy
=====

`Galpy <https://docs.galpy.org/en/>`_ is another popular Python package for
Galactic dynamics with similar functionality to Gala. For example, Galpy
supports creating gravitational potential objects and numerically integrating
orbits (among other things).

Gala provides an interface for converting representations of orbits from Gala to
Galpy, or from Galpy to Gala. To convert a Gala :class:`~gala.dynamics.Orbit`
object to a Galpy ``Orbit``, use the
:meth:`~gala.dynamics.Orbit.to_galpy_orbit()` method:

.. doctest-requires:: galpy

    >>> w0 = gd.PhaseSpacePosition(pos=[10., 0, 0] * u.kpc,
    ...                            vel=[0, 0, 200.] * u.km/u.s)
    >>> mw = gp.Hamiltonian(gp.MilkyWayPotential())
    >>> orbit = mw.integrate_orbit(w0, dt=1, n_steps=1000)
    >>> orbit
    <Orbit cartesian, dim=3, shape=(1001,)>
    >>> galpy_orbit = orbit.to_galpy_orbit()
    >>> galpy_orbit  # doctest: +SKIP
    <galpy.orbit.Orbits.Orbit object at 0x7f99b3c8fe80>

Similarly, a Galpy ``Orbit`` can be converted to a Gala
:class:`~gala.dynamics.Orbit` using the
:meth:`~gala.dynamics.Orbit.from_galpy_orbit()` classmethod:

.. doctest-requires:: galpy

    >>> import galpy.potential as galpy_p
    >>> import galpy.orbit as galpy_o
    >>> mp = galpy_p.MiyamotoNagaiPotential(a=0.5, b=0.0375, amp=1.,
    ...                                     normalize=1.)
    >>> galpy_orbit = galpy_o.Orbit([1., 0.1, 1.1, 0., 0.1, 1.])
    >>> ts = np.linspace(0, 100, 10000)
    >>> galpy_orbit.integrate(ts, mp, method='odeint')
    >>> orbit = gd.Orbit.from_galpy_orbit(galpy_orbit)
    >>> orbit
    <Orbit cylindrical, dim=3, shape=(10000,)>

Gala also provides tools for converting potential objects to `galpy` potential
objects, or creating Gala potential objects from existing `galpy` potentials.
To convert a Gala potential to a Galpy potential, use the
:meth:`~gala.potential.potential.PotentialBase.to_galpy_potential()` method on
any Gala potential object. For example:

.. doctest-requires:: galpy

    >>> pot = gp.HernquistPotential(m=1e10*u.Msun, c=1.5*u.kpc, units=galactic)
    >>> galpy_pot = pot.to_galpy_potential()
    >>> galpy_pot  # doctest: +SKIP
    <galpy.potential.TwoPowerSphericalPotential.HernquistPotential at 0x7faa00263b20>
    >>> galpy_pot.Rforce(1., 0.)  # doctest: +FLOAT_CMP
    -0.48737954713808573

To convert from a Galpy potential to a Gala potential, use the
:func:`~gala.potential.potential.interop.galpy_to_gala_potential()` function. For
example:

.. doctest-requires:: galpy

    >>> import galpy.potential as galpy_gp
    >>> from gala.potential.potential.interop import galpy_to_gala_potential
    >>> galpy_pot = galpy_gp.HernquistPotential(amp=1., a=0.5)
    >>> pot = galpy_to_gala_potential(galpy_pot)
    >>> pot
    <HernquistPotential: m=4.50e+10, c=4.00 (kpc,Myr,solMass,rad)>


Agama
=====

Coming soon, but we could use your help! Please leave a note `in this issue
<https://github.com/adrn/gala/issues/230>`_ if you would find interoperability
with Agama useful.
