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

Gala provides tools for converting potential objects to `galpy` potential
objects, or creating Gala potential objects from existing `galpy` potentials.
To convert a Gala potential to a Galpy potential, use the
:meth:`~gala.potential.potential.PotentialBase.as_interop()` method on any Gala
potential object with ``"galpy"``. For example:

.. doctest-requires:: galpy

    >>> pot = gp.HernquistPotential(m=1e10 * u.Msun, c=1.5 * u.kpc, units=galactic)
    >>> galpy_pot = pot.as_interop("galpy")
    >>> galpy_pot  # doctest: +SKIP
    <galpy.potential.TwoPowerSphericalPotential.HernquistPotential at 0x7faa00263b20>
    >>> float(galpy_pot.Rforce(1., 0.))  # doctest: +FLOAT_CMP
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
    <HernquistPotential: m=4.50e+10 solMass, c=4.00 kpc (kpc,Myr,solMass,rad)>


Agama
=====

`Agama <https://github.com/GalacticDynamics-Oxford/Agama>`_ is a library for
modeling galaxies, with fast potential expansions and tools for distribution
functions and actions. To convert a Gala potential to an Agama potential, use
:meth:`~gala.potential.potential.PotentialBase.as_interop()` with ``"agama"``:

.. doctest-requires:: agama

    >>> pot = gp.HernquistPotential(m=1e10 * u.Msun, c=1.5 * u.kpc, units=galactic)
    >>> agama_pot = pot.as_interop("agama")
    >>> float(agama_pot.potential([1., 0, 0]))  # doctest: +FLOAT_CMP
    -0.017994008608459656
    >>> pot.energy([1., 0, 0] * u.kpc)  # doctest: +FLOAT_CMP
    <Quantity [-0.01799401] kpc2 / Myr2>

Note that this calls ``agama.setUnits()`` with the unit system of the Gala
potential, which changes Agama's units globally.

Gala can also load an Agama ``CylSpline`` potential export file with
:meth:`~gala.potential.potential.CylSplinePotential.from_file()`.
