import operator

import astropy.coordinates as coord
import astropy.units as u
import numpy as np

from gala._compat_utils import COPY_IF_NEEDED

__all__ = ["NDCartesianDifferential", "NDCartesianRepresentation"]


def _make_getter(component):
    """Make an attribute getter for use in a property.

    Removed from Astropy in v6.3 but still used here.

    Parameters
    ----------
    component : str
        The name of the component that should be accessed.  This assumes the
        actual value is stored in an attribute of that name prefixed by '_'.
    """
    # This has to be done in a function to ensure the reference to component
    # is not lost/redirected.
    component = "_" + component

    def get_component(self):
        return getattr(self, component)

    return get_component


class NDMixin:
    def _apply(self, method, *args, **kwargs):
        """Create a new representation with ``method`` applied to the arrays.

        In typical usage, the method is any of the shape-changing methods for
        `~numpy.ndarray` (``reshape``, ``swapaxes``, etc.), as well as those
        picking particular elements (``__getitem__``, ``take``, etc.), which
        are all defined in `~astropy.utils.misc.ShapedLikeNDArray`. It will be
        applied to the underlying arrays (e.g., ``x``, ``y``, and ``z`` for
        `~astropy.coordinates.CartesianRepresentation`), with the results used
        to create a new instance.

        Internally, it is also used to apply functions to the components
        (in particular, `~numpy.broadcast_to`).

        Parameters
        ----------
        method : str or callable
            If str, it is the name of a method that is applied to the internal
            ``components``. If callable, the function is applied.
        args : tuple
            Any positional arguments for ``method``.
        kwargs : dict
            Any keyword arguments for ``method``.
        """
        if callable(method):
            apply_method = lambda array: method(array, *args, **kwargs)
        else:
            apply_method = operator.methodcaller(method, *args, **kwargs)
        return self.__class__(
            [apply_method(getattr(self, component)) for component in self.components],
            copy=COPY_IF_NEEDED,
        )


class NDCartesianRepresentation(NDMixin, coord.CartesianRepresentation):
    """
    Representation of points in ND cartesian coordinates.

    Parameters
    ----------
    x : `~astropy.units.Quantity` or array
        The Cartesian coordinates of the point(s). If not quantity,
        ``unit`` should be set.
    differentials : dict, `NDCartesianDifferential` (optional)
        Any differential classes that should be associated with this
        representation.
    unit : `~astropy.units.Unit` or str
        If given, the coordinates will be converted to this unit (or taken to
        be in this unit if not given.
    copy : bool, optional
        If `True` (default), arrays will be copied rather than referenced.
    """

    attr_classes = {}

    def __init__(self, x, differentials=None, unit=None, copy=True):
        if unit is None:
            unit = u.one if not hasattr(x[0], "unit") else x[0].unit

        x = u.Quantity(x, unit, copy=copy, subok=True)
        copy = False

        self.attr_classes = {"x" + str(i): u.Quantity for i in range(1, len(x) + 1)}

        super(coord.CartesianRepresentation, self).__init__(
            *x, differentials=differentials, copy=copy
        )

        ptype = None
        for name, _ in self.attr_classes.items():
            if ptype is None:
                ptype = getattr(self, "_" + name).unit.physical_type

            elif getattr(self, "_" + name).unit.physical_type != ptype:
                raise u.UnitsError("All components should have matching physical types")

            cls = self.__class__
            if not hasattr(cls, name):
                setattr(
                    cls,
                    name,
                    property(
                        _make_getter(name),
                        doc=(f"The '{name}' component of the points(s)."),
                    ),
                )

    @property
    def masked(self):
        """NOTE: This overrides the support for masks"""
        return False

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
            msg = f"xyz_axis {xyz_axis} out of bounds [-{result_ndim}, {result_ndim})"
            raise IndexError(msg)
        if xyz_axis < 0:
            xyz_axis += result_ndim

        # Get components to the same units (very fast for identical units)
        # since np.concatenate cannot deal with quantity.
        unit = self._x1.unit

        sh = self.shape
        sh = (*sh[:xyz_axis], 1, *sh[xyz_axis:])
        components = [
            getattr(self, "_" + name).reshape(sh).to(unit).value
            for name in self.attr_classes
        ]
        xs_value = np.concatenate(components, axis=xyz_axis)
        return u.Quantity(xs_value, unit=unit, copy=COPY_IF_NEEDED)

    xyz = property(get_xyz)


class NDCartesianDifferential(NDMixin, coord.CartesianDifferential):
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
    attr_classes = {}

    def __init__(self, d_x, unit=None, copy=True):
        if unit is None:
            unit = u.one if not hasattr(d_x[0], "unit") else d_x[0].unit

        d_x = u.Quantity(d_x, unit, copy=copy, subok=True)
        copy = False

        self.attr_classes = {"d_x" + str(i): u.Quantity for i in range(1, len(d_x) + 1)}

        super(coord.CartesianDifferential, self).__init__(*d_x, copy=copy)

        ptype = None
        for name, _ in self.attr_classes.items():
            if ptype is None:
                ptype = getattr(self, "_" + name).unit.physical_type

            elif getattr(self, "_" + name).unit.physical_type != ptype:
                raise u.UnitsError("All components should have matching physical types")

            cls = self.__class__
            if not hasattr(cls, name):
                setattr(
                    cls,
                    name,
                    property(
                        _make_getter(name),
                        doc=(f"The '{name}' component of the points(s)."),
                    ),
                )

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
            msg = f"xyz_axis {xyz_axis} out of bounds [-{result_ndim}, {result_ndim})"
            raise IndexError(msg)
        if xyz_axis < 0:
            xyz_axis += result_ndim

        # Get components to the same units (very fast for identical units)
        # since np.concatenate cannot deal with quantity.
        unit = self._d_x1.unit

        sh = self.shape
        sh = (*sh[:xyz_axis], 1, *sh[xyz_axis:])
        components = [
            getattr(self, "_" + name).reshape(sh).to(unit).value
            for name in self.components
        ]
        xs_value = np.concatenate(components, axis=xyz_axis)
        return u.Quantity(xs_value, unit=unit, copy=COPY_IF_NEEDED)

    d_xyz = property(get_d_xyz)
