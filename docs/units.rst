.. include:: references.txt

.. _gala-units:

***************************
Unit Systems (`gala.units`)
***************************

Introduction
============

This module contains a class for handling systems of units, and provides a few
pre-defined unit systems that are useful for galactic dynamics.

For the examples below, I assume the following imports have already been
executed::

    >>> import astropy.units as u
    >>> import numpy as np
    >>> from gala.units import UnitSystem

Unit Systems
============

A unit system is defined by a set of base units that specify length, time, mass,
and angle units. A `~gala.units.UnitSystem` object is created by passing in
units with (at least) these four required physical types::

    >>> usys = UnitSystem(u.cm, u.millisecond, u.degree, u.gram)
    >>> usys
    <UnitSystem (cm, ms, g, deg)>

Astropy :class:`~astropy.units.Quantity` objects can be decomposed into this
unit system using :meth:`~astropy.units.Quantity.decompose`::

    >>> a = 15 * u.km/u.s
    >>> a.decompose(usys)
    <Quantity 1500. cm / ms>

`~gala.units.UnitSystem` objects can also act as a dictionary to look up a unit
for a given physical type. For example, if we want to know what a 'velocity'
unit is in a given unit system, pass the key ``'speed'`` or ``'velocity'``::

    >>> usys['speed']
    Unit("cm / ms")

This works for the base unit physical types and for more complex physical
types::

    >>> usys['length']
    Unit("cm")
    >>> usys['pressure']
    Unit("g / (cm ms2)")

In Astropy version 4.3 and later, units from `~gala.units.UnitSystem` objects
can also be retrieved by passing in Astropy ``PhysicalType`` instances as keys,
for example:

.. doctest-requires:: astropy>=4.3

    >>> ptype = u.get_physical_type('length')**2 / u.get_physical_type('time')
    >>> usys[ptype]
    Unit("cm2 / ms")


Creating unit systems with scaled base units
--------------------------------------------

It is sometimes useful to construct a unit system with base units that are
scaled versions of units. For example, you may want to create a unit system with
the base units (10 kpc, 200 Myr, 1000 Msun). To construct a
`~gala.units.UnitSystem` with scaled base units, pass in
`~astropy.units.Quantity` objects. For example::

    >>> usys = UnitSystem(10 * u.kpc, 200 * u.Myr, 1000 * u.Msun, u.radian)
    >>> usys
    <UnitSystem (10.0 kpc, 200.0 Myr, 1000.0 solMass, rad)>
    >>> q = 15.7 * u.kpc
    >>> q.decompose(usys)
    <Quantity 1.57 10.0 kpc>

Or, to create a unit system in which G=1, given length and mass units::

    >>> from astropy.constants import G
    >>> L_unit = 1 * u.kpc
    >>> M_unit = 1e6 * u.Msun
    >>> T_unit = np.sqrt((L_unit**3) / (G * M_unit))
    >>> usys = UnitSystem(L_unit, M_unit, T_unit.to(u.Myr), u.radian)
    >>> np.round(usys.get_constant('G'), 5)  # doctest: +FLOAT_CMP
    1.0


Custom display units
--------------------

It is sometimes useful to have default display units for physical types that are
not simple compositions of base units. For example, for kinematics within the
Milky Way, a common base unit system consists of (kpc, Myr, Msun), but
velocities are often expressed or displayed in km/s. To change the default
display unit of a composite unit, specify the preferred unit on creation::

    >>> usys = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun)
    >>> usys2 = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun, u.km/u.s)
    >>> usys['velocity'], usys2['velocity']
    (Unit("kpc / Myr"), Unit("km / s"))

For unit systems with specified composite units (e.g., ``usys2`` above),
the Astropy :meth:`~astropy.units.Quantity.decompose` method will fail because
it only uses the base units::

    >>> q = 150 * u.pc/u.Myr
    >>> q.decompose(usys2)
    <Quantity 0.15 kpc / Myr>

Because we specified a unit for quantities with a physical type = 'velocity', we
can instead use the `~gala.units.UnitSystem.decompose` method of the
`~gala.units.UnitSystem` object to retrieve the object in the desired display
unit::

    >>> usys2.decompose(q)
    <Quantity 146.66883325 km / s>


.. _gala-units-api:

API
===

.. automodapi:: gala.units
    :no-inheritance-diagram:
