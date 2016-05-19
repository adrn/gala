.. include:: references.txt

.. _units:

***************************
Unit Systems (`gala.units`)
***************************

Introduction
============

This subpackage contains a class for handling systems of units and has
a few useful, pre-defined systems for astrodynamics.

For the examples below, I assume the following imports have been
already executed::

    >>> import astropy.units as u
    >>> from gala.units import UnitSystem

Unit Systems
============

.. warning::

    Unit system support may eventually move to the Astropy project.

At its simplest, a unit system is a container class that holds a
set of base units that specify length, time, mass, and angle. A
`~gala.units.UnitSystem` object is created by passing in units with
(at least) these four required physical types::

    >>> usys = UnitSystem(u.cm, u.millisecond, u.degree, u.gram)
    >>> usys
    <UnitSystem (cm,ms,g,deg)>

Astropy :class:`~astropy.units.Quantity` objects can be decomposed into this
unit system using :meth:`~astropy.units.Quantity.decompose`::

    >>> a = 15*u.km/u.s
    >>> a.decompose(usys)
    <Quantity 1500.0 cm / ms>

`~gala.units.UnitSystem` objects can also act as a dictionary to look up
a unit for a given physical type. For example, if we want to know what a
'speed' unit is in a given unit system, simple pass ``'speed'`` in as a key::

    >>> usys['speed']
    Unit("cm / ms")

This works for the base unit physical types as well, and for more complex
physical types::

    >>> usys['length']
    Unit("cm")
    >>> usys['pressure']
    Unit("g / (cm ms2)")

Default representations of composite units
------------------------------------------

It is sometimes useful to have default representations for physical types
that are not composites of base units. For example, for kinematic within
the Milky Way, the base unit system is usually kpc, Myr, Msun, radian, but
velocities (speeds) are often given in terms of km/s. To change the default
representation of a composite unit, just specify the preferred unit on
creation::

    >>> usys = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun)
    >>> usys2 = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun, u.km/u.s)
    >>> usys['speed'], usys2['speed']
    (Unit("kpc / Myr"), Unit("km / s"))

For unit systems with specified composite units (e.g., ``usys2`` above),
the Astropy :meth:`~astropy.units.Quantity.decompose` method will fail because
it only uses the base units::

    >>> q = (150*u.pc/u.Myr)
    >>> q.decompose(usys2)
    <Quantity 0.15 kpc / Myr>

Because we specified a unit for quantities with a physical type = 'speed',
we instead want to use the `~gala.units.UnitSystem.decompose` method
of the `~gala.units.UnitSystem` object, which has exactly opposite
call syntax::

    >>> usys2.decompose(q)
    <Quantity 146.66883325096924 km / s>

.. _units-api:

API
===

.. automodapi:: gala.units
    :no-inheritance-diagram:

