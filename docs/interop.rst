.. _gala-interop:

*********************************************
Interoperability with Other Dynamics Packages
*********************************************


Galpy
=====

TODO: convert orbits to/from Galpy orbits

Gala also provides tools for converting potential objects to `galpy` potential
objects, or creating Gala potential objects from existing `galpy` potentials.
To convert a Gala potential to a Galpy potential, use the
:meth:`PotentialBase.to_galpy_potential()` method. For example::

    >>> import astropy.units as u
    >>> import gala.potential as gp
    >>> from gala.units import galactic
    >>> pot = gp.HernquistPotential(m=1e10*u.Msun, c=1.5*u.kpc, units=galactic)
    >>> galpy_pot = pot.to_galpy_potential()
    >>> galpy_pot
    <galpy.potential.TwoPowerSphericalPotential.HernquistPotential at 0x7faa00263b20>
    >>> galpy_pot.Rforce(1., 0.)  # doctest: +FLOAT_CMP
    -1.4598592245082576

To convert from a Galpy potential to a Gala potential, use the
:func:`gala.potential.galpy_to_gala_potential()` function. For example::

    >>> import galpy.potential as galpy_gp
    >>> galpy_pot = galpy_gp.HernquistPotential(amp=1., a=0.5)
    >>> pot = gp.galpy_to_gala_potential(galpy_pot)
    >>> pot
    <HernquistPotential: m=4.50e+10, c=4.00 (kpc,Myr,solMass,rad)>
