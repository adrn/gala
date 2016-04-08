# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
from astropy.units.physical import _physical_unit_mapping
import astropy.constants as const

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

    def __init__(self, units, *args):

        self._required_physical_types = ['length', 'time', 'mass', 'angle']
        self._core_units = []

        if isinstance(units, UnitSystem):
            self._registry = units._registry.copy()
            self._core_units = units._core_units
            return

        if len(args) > 0:
            units = (units,) + tuple(args)

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
        """
        Return a dictionary representation of the unit system with keys
        set by the physical types and values set by the unit objects.
        """
        return self._registry.copy()

    def decompose(self, q):
        """
        A thin wrapper around :meth:`astropy.units.Quantity.decompose` that
        knows how to handle Quantities with physical types with non-default
        representations.

        Parameters
        ----------
        q : :class:`~astropy.units.Quantity`
            An instance of an astropy Quantity object.

        Returns
        -------
        q : :class:`~astropy.units.Quantity`
            A new quantity, decomposed to represented in this unit system.
        """
        try:
            ptype = q.unit.physical_type
        except AttributeError:
            raise TypeError("Object must be an astropy.units.Quantity, not "
                            "a '{}'.".format(q.__class__.__name__))

        if ptype in self._registry:
            return q.to(self._registry[ptype])
        else:
            return q.decompose(self)

    def get_constant(self, name):
        """
        Retrieve a constant with specified name in this unit system.

        Parameters
        ----------
        name : str
            The name of the constant, e.g., G.

        Returns
        -------
        const : float
            The value of the constant represented in this unit system.

        Examples
        --------

            >>> usys = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun)
            >>> usys.get_constant('c')
            306.6013937879527

        """
        try:
            c = getattr(const, name)
        except AttributeError:
            raise ValueError("Constant name '{}' doesn't exist in astropy.constants".format(name))

        return c.decompose(self._core_units).value

class DimensionlessUnitSystem(UnitSystem):

    def __init__(self):
        self._core_units = [u.one]
        self._registry = dict()
        self._registry['dimensionless'] = u.one

    def __getitem__(self, key):
        return u.one

    def __str__(self):
        return "UnitSystem (dimensionless)"

    def to_dict(self):
        raise ValueError("Cannot represent dimensionless unit system as dict!")

    def get_constant(self, name):
        raise ValueError("Cannot get constant in dimensionless units!")

# define galactic unit system
galactic = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian,
                      u.km/u.s, u.mas/u.yr)

# solar system units
solarsystem = UnitSystem(u.au, u.M_sun, u.yr, u.radian)

# dimensionless
dimensionless = DimensionlessUnitSystem()
