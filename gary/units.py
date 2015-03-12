# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
from astropy.units.physical import _physical_unit_mapping

class UnitSystem(object):
    """

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

            return unit.decompose(self._registry.values())

    def __len__(self):
        return len(self._core_units)

    def __iter__(self):
        for uu in self._core_units:
            yield uu

    def __str__(self):
        return "UnitSystem ({0})".format(",".join([str(uu) for uu in self._core_units]))

    def __repr__(self):
        return "<{0}>".format(self.__str__())

# define galactic unit system
galactic = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian,
                      u.km/u.s, u.mas/u.yr)
