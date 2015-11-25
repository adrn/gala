# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
from astropy.units.physical import _physical_unit_mapping

class UnitSystem(object):
    """
    Represents a system of units. At minimum, this consists of a set of
    length, time, mass, and angle units, but may also contain preferred
    representations for composite units. For example, the base unit system
    could be ``{kpc, Myr, Msun, radian}``, but you can also specify a preferred
    speed, such as ``km/s``.

    This class functions like a dictionary with keys set by physical types.
    If a unit for a particular physical type is not specified on creation,
    a composite unit will be created with the base units. See Examples below
    for some demonstrations.

    Parameters
    ----------
    *units
        The units that define the unit system. At minimum, this must
        contain length, time, mass, and angle units.

    Examples
    --------
    If only base units are specified, any physical type specified as a key
    to this object will be composed out of the base units::

        >>> usys = UnitSystem(u.m, u.s, u.kg, u.radian)
        >>> usys['energy']
        Unit("kg m2 / s2")

    However, custom representations for composite units can also be specified
    when initializing::

        >>> usys = UnitSystem(u.m, u.s, u.kg, u.radian, u.erg)
        >>> usys['energy']
        Unit("erg")

    This is useful for Galactic dynamics where lengths and times are usually
    given in terms of ``kpc`` and ``Myr``, but speeds are given in ``km/s``::

        >>> usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian, u.km/u.s)
        >>> usys['speed']
        Unit("km / s")

    """
    def __init__(self, *units):

        self._required_physical_types = ['length', 'time', 'mass', 'angle']
        self._core_units = []

        self._registry = dict()
        for unit in units:
            typ = unit.physical_type
            if typ in self._registry:
                raise ValueError("Multiple units passed in with type '{0}'".format(typ))
            self._registry[typ] = unit

        for phys_type in self._required_physical_types:
            if phys_type not in self._registry:
                raise ValueError("You must specify a unit with physical type '{0}'".format(phys_type))
            self._core_units.append(self._registry[phys_type])

    def __getitem__(self, key):

        if key in self._registry:
            return self._registry[key]

        else:
            unit = None
            for k,v in _physical_unit_mapping.items():
                if v == key:
                    unit = u.Unit(" ".join(["{}**{}".format(x,y) for x,y in k]))
                    break

            if unit is None:
                raise ValueError("Physical type '{0}' doesn't exist in unit registry.".format(key))

            unit = unit.decompose(self._core_units)
            unit._scale = 1.
            return unit

    def __len__(self):
        return len(self._core_units)

    def __iter__(self):
        for uu in self._core_units:
            yield uu

    def __str__(self):
        return "UnitSystem ({0})".format(",".join([str(uu) for uu in self._core_units]))

    def __repr__(self):
        return "<{0}>".format(self.__str__())

    def to_dict(self):
        return self._registry.copy()

# define galactic unit system
galactic = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian,
                      u.km/u.s, u.mas/u.yr)

# solar system units
solarsystem = UnitSystem(u.au, u.M_sun, u.yr, u.radian)
