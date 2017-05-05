# coding: utf-8

from __future__ import division, print_function

# Standard library
from collections import OrderedDict

# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np

__all__ = ['NDCartesianRepresentation', 'NDCartesianDifferential']

class NDCartesianRepresentation(coord.CartesianRepresentation):
    """
    Representation of points in ND cartesian coordinates.

    Parameters
    ----------
    x : `~astropy.units.Quantity` or array
        The Cartesian coordinates of the point(s). If not quantity,
        ``unit`` should be set.
    unit : `~astropy.units.Unit` or str
        If given, the coordinates will be converted to this unit (or taken to
        be in this unit if not given.
    copy : bool, optional
        If `True` (default), arrays will be copied rather than referenced.
    """

    attr_classes = OrderedDict()

    def __init__(self, *x, unit=None, copy=True):

        if not x:
            raise ValueError('You must pass in at least 1D data.')

        if unit is None:
            if not hasattr(x[0], 'unit'):
                unit = u.one
            else:
                unit = x[0].unit

        x = u.Quantity(x, unit, copy=copy, subok=True)
        copy = False

        self.attr_classes = OrderedDict([('x'+str(i), u.Quantity)
                                         for i in range(1, len(x)+1)])

        super(coord.CartesianRepresentation, self).__init__(*x, copy=copy)

        ptype = None
        for name,_ in self.attr_classes.items():
            if ptype is None:
                ptype = getattr(self, '_'+name).unit.physical_type

            else:
                if getattr(self, '_'+name).unit.physical_type != ptype:
                    raise u.UnitsError("All components should have matching "
                                       "physical types")

            cls = self.__class__
            if not hasattr(cls, name):
                setattr(cls, name,
                        property(coord.representation._make_getter(name),
                                 doc=("The '{0}' component of the points(s)."
                                      .format(name))))

    def get_xyz(self, xyz_axis=0):
        """Return a vector array of the x, y, and z coordinates.

        Parameters
        ----------
        xyz_axis : int, optional
            The axis in the final array along which the x, y, z components
            should be stored (default: 0).

        Returns
        -------
        xs : `~astropy.units.Quantity`
            With dimension 3 along ``xyz_axis``.
        """
        # Add new axis in x, y, z so one can concatenate them around it.
        # NOTE: just use np.stack once our minimum numpy version is 1.10.
        result_ndim = self.ndim + 1
        if not -result_ndim <= xyz_axis < result_ndim:
            raise IndexError('xyz_axis {0} out of bounds [-{1}, {1})'
                             .format(xyz_axis, result_ndim))
        if xyz_axis < 0:
            xyz_axis += result_ndim

        # Get components to the same units (very fast for identical units)
        # since np.concatenate cannot deal with quantity.
        unit = self._x1.unit

        sh = self.shape
        sh = sh[:xyz_axis] + (1,) + sh[xyz_axis:]
        components = [getattr(self, '_'+name).reshape(sh).to(unit).value
                      for name in self.attr_classes]
        xs_value = np.concatenate(components, axis=xyz_axis)
        return u.Quantity(xs_value, unit=unit, copy=False)

    xyz = property(get_xyz)

class NDCartesianDifferential(coord.CartesianDifferential):
    """Differentials in of points in ND cartesian coordinates.

    Parameters
    ----------
    *d_x : `~astropy.units.Quantity` or array
        The Cartesian coordinates of the differentials. If not quantity,
        ``unit`` should be set.
    unit : `~astropy.units.Unit` or str
        If given, the differentials will be converted to this unit (or taken to
        be in this unit if not given.
    copy : bool, optional
        If `True` (default), arrays will be copied rather than referenced.
    """
    base_representation = NDCartesianRepresentation
    attr_classes = OrderedDict()

    def __init__(self, *d_x, unit=None, copy=True):

        if not d_x:
            raise ValueError('You must pass in at least 1D data.')

        if unit is None:
            if not hasattr(d_x[0], 'unit'):
                unit = u.one
            else:
                unit = d_x[0].unit

        d_x = u.Quantity(d_x, unit, copy=copy, subok=True)
        copy = False

        self.attr_classes = OrderedDict([('d_x'+str(i), u.Quantity)
                                         for i in range(1, len(d_x)+1)])

        super(coord.CartesianDifferential, self).__init__(*d_x, copy=copy)

        ptype = None
        for name,_ in self.attr_classes.items():
            if ptype is None:
                ptype = getattr(self, '_'+name).unit.physical_type

            else:
                if getattr(self, '_'+name).unit.physical_type != ptype:
                    raise u.UnitsError("All components should have matching "
                                       "physical types")

            cls = self.__class__
            if not hasattr(cls, name):
                setattr(cls, name,
                        property(coord.representation._make_getter(name),
                                 doc=("The '{0}' component of the points(s)."
                                      .format(name))))

    def get_d_xyz(self, xyz_axis=0):
        """Return a vector array of the x, y, and z coordinates.

        Parameters
        ----------
        xyz_axis : int, optional
            The axis in the final array along which the x, y, z components
            should be stored (default: 0).

        Returns
        -------
        d_xs : `~astropy.units.Quantity`
            With dimension 3 along ``xyz_axis``.
        """
        # Add new axis in x, y, z so one can concatenate them around it.
        # NOTE: just use np.stack once our minimum numpy version is 1.10.
        result_ndim = self.ndim + 1
        if not -result_ndim <= xyz_axis < result_ndim:
            raise IndexError('xyz_axis {0} out of bounds [-{1}, {1})'
                             .format(xyz_axis, result_ndim))
        if xyz_axis < 0:
            xyz_axis += result_ndim

        # Get components to the same units (very fast for identical units)
        # since np.concatenate cannot deal with quantity.
        unit = self._d_x1.unit

        sh = self.shape
        sh = sh[:xyz_axis] + (1,) + sh[xyz_axis:]
        components = [getattr(self, '_'+name).reshape(sh).to(unit).value
                      for name in self.components]
        xs_value = np.concatenate(components, axis=xyz_axis)
        return u.Quantity(xs_value, unit=unit, copy=False)

    d_xyz = property(get_d_xyz)
