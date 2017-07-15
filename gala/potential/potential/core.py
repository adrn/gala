# coding: utf-8

from __future__ import division, print_function


# Standard library
import abc
from collections import OrderedDict
import warnings
import uuid

# Third-party
import numpy as np
from astropy.constants import G
import astropy.units as u
from astropy.utils import isiterable
import six

# Project
from ..common import CommonBase
from ...dynamics import PhaseSpacePosition
from ...util import ImmutableDict, atleast_2d
from ...units import DimensionlessUnitSystem

__all__ = ["PotentialBase", "CompositePotential"]

@six.add_metaclass(abc.ABCMeta)
class PotentialBase(CommonBase):
    """
    A baseclass for defining pure-Python gravitational potentials.

    Subclasses must define (at minimum) a method that evaluates
    the potential energy at a given position ``q``
    and time ``t``: ``_energy(q, t)``. For integration, the subclasses
    must also define a method to evaluate the gradient,
    ``_gradient(q,t)``. Optionally, they may also define methods
    to compute the density and hessian: ``_density()``, ``_hessian()``.
    """

    def __init__(self, parameters, origin=None, parameter_physical_types=None,
                 ndim=3, units=None):

        self.units = self._validate_units(units)

        if parameter_physical_types is None:
            parameter_physical_types = dict()
        self._ptypes = parameter_physical_types
        self.parameters = self._prepare_parameters(parameters, self._ptypes,
                                                   self.units)

        try:
            self.G = G.decompose(self.units).value
        except u.UnitConversionError:
            self.G = 1. # TODO: this is a HACK and could lead to user confusion

        self.ndim = ndim

        if origin is None:
            origin = np.zeros(self.ndim)
        self.origin = self._remove_units(origin)

    # ========================================================================
    # Abstract methods that must be implemented by subclasses
    #
    @abc.abstractmethod
    def _energy(self, q, t=0.):
        pass

    @abc.abstractmethod
    def _gradient(self, q, t=0.):
        pass

    def _density(self, q, t=0.):
        raise NotImplementedError("This Potential has no implemented density function.")

    def _hessian(self, q, t=0.):
        raise NotImplementedError("This Potential has no implemented Hessian.")

    # ========================================================================
    # Utility methods
    #
    def _remove_units(self, x):
        """
        Always returns an array. If a Quantity is passed in, it converts to the
        units associated with this object and returns the value.
        """
        if hasattr(x, 'unit'):
            x = x.decompose(self.units).value

        else:
            x = np.array(x)

        return x

    def _remove_units_prepare_shape(self, x):
        """
        This is similar to that implemented by `gala.potential.common.CommonBase`,
        but returns just the position if the input is a `PhaseSpacePosition`.
        """
        if hasattr(x, 'unit'):
            x = x.decompose(self.units).value

        elif isinstance(x, PhaseSpacePosition):
            x = x.cartesian.xyz.decompose(self.units).value

        x = atleast_2d(x, insert_axis=1).astype(np.float64)
        return x

    # ========================================================================
    # Core methods that use the above implemented functions
    #
    def energy(self, q, t=0.):
        """
        Compute the potential energy at the given position(s).

        Parameters
        ----------
        q : `~gala.dynamics.PhaseSpacePosition`, `~astropy.units.Quantity`, array_like
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.

        Returns
        -------
        E : `~astropy.units.Quantity`
            The potential energy per unit mass or value of the potential.
        """
        q = self._remove_units_prepare_shape(q)
        orig_shape,q = self._get_c_valid_arr(q)
        t = self._validate_prepare_time(t, q)
        ret_unit = self.units['energy'] / self.units['mass']

        return self._energy(q, t=t).T.reshape(orig_shape[1:]) * ret_unit

    def gradient(self, q, t=0.):
        """
        Compute the gradient of the potential at the given position(s).

        Parameters
        ----------
        q : `~gala.dynamics.PhaseSpacePosition`, `~astropy.units.Quantity`, array_like
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.

        Returns
        -------
        grad : `~astropy.units.Quantity`
            The gradient of the potential. Will have the same shape as
            the input position.
        """
        q = self._remove_units_prepare_shape(q)
        orig_shape,q = self._get_c_valid_arr(q)
        t = self._validate_prepare_time(t, q)
        ret_unit = self.units['length'] / self.units['time']**2
        return (self._gradient(q, t=t).T.reshape(orig_shape) * ret_unit).to(self.units['acceleration'])

    def density(self, q, t=0.):
        """
        Compute the density value at the given position(s).

        Parameters
        ----------
        q : `~gala.dynamics.PhaseSpacePosition`, `~astropy.units.Quantity`, array_like
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.

        Returns
        -------
        dens : `~astropy.units.Quantity`
            The potential energy or value of the potential. If the input
            position has shape ``q.shape``, the output energy will have
            shape ``q.shape[1:]``.
        """
        q = self._remove_units_prepare_shape(q)
        orig_shape,q = self._get_c_valid_arr(q)
        t = self._validate_prepare_time(t, q)
        ret_unit = self.units['mass'] / self.units['length']**3
        return (self._density(q, t=t).T * ret_unit).to(self.units['mass density'])

    def hessian(self, q, t=0.):
        """
        Compute the Hessian of the potential at the given position(s).

        Parameters
        ----------
        q : `~gala.dynamics.PhaseSpacePosition`, `~astropy.units.Quantity`, array_like
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.

        Returns
        -------
        hess : `~astropy.units.Quantity`
            The Hessian matrix of second derivatives of the potential. If the input
            position has shape ``q.shape``, the output energy will have shape
            ``(q.shape[0],q.shape[0]) + q.shape[1:]``. That is, an ``n_dim`` by
            ``n_dim`` array (matrix) for each position.
        """
        q = self._remove_units_prepare_shape(q)
        orig_shape,q = self._get_c_valid_arr(q)
        t = self._validate_prepare_time(t, q)
        ret_unit = 1 / self.units['time']**2
        hess = np.moveaxis(self._hessian(q, t=t), 0, -1)
        return hess.reshape((orig_shape[0], orig_shape[0]) + orig_shape[1:]) * ret_unit

    # ========================================================================
    # Convenience methods that make use the base methods
    #
    def acceleration(self, q, t=0.):
        """
        Compute the acceleration due to the potential at the given position(s).

        Parameters
        ----------
        q : `~gala.dynamics.PhaseSpacePosition`, `~astropy.units.Quantity`, array_like
            Position to compute the acceleration at.

        Returns
        -------
        acc : `~astropy.units.Quantity`
            The acceleration. Will have the same shape as the input
            position array, ``q``.
        """
        return -self.gradient(q, t=t)

    def mass_enclosed(self, q, t=0.):
        """
        Estimate the mass enclosed within the given position by assuming the potential
        is spherical.

        Parameters
        ----------
        q : `~gala.dynamics.PhaseSpacePosition`, `~astropy.units.Quantity`, array_like
            Position(s) to estimate the enclossed mass.

        Returns
        -------
        menc : `~astropy.units.Quantity`
            Mass enclosed at the given position(s). If the input position
            has shape ``q.shape``, the output energy will have shape
            ``q.shape[1:]``.
        """
        q = self._remove_units_prepare_shape(q)
        orig_shape,q = self._get_c_valid_arr(q)
        t = self._validate_prepare_time(t, q)

        # small step-size in direction of q
        h = 1E-3 # MAGIC NUMBER

        # Radius
        r = np.sqrt(np.sum(q**2, axis=1))

        epsilon = h*q/r[:,np.newaxis]

        dPhi_dr_plus = self._energy(q + epsilon, t=t)
        dPhi_dr_minus = self._energy(q - epsilon, t=t)
        diff = (dPhi_dr_plus - dPhi_dr_minus)

        if isinstance(self.units, DimensionlessUnitSystem):
            Gee = 1.
        else:
            Gee = G.decompose(self.units).value

        Menc = np.abs(r*r * diff / Gee / (2.*h))
        Menc = Menc.reshape(orig_shape[1:])

        return Menc * self.units['mass']

    def circular_velocity(self, q, t=0.):
        """
        Estimate the circular velocity at the given position assuming the potential
        is spherical.

        Parameters
        ----------
        q : array_like, numeric
            Position(s) to estimate the circular velocity.

        Returns
        -------
        vcirc : `~astropy.units.Quantity`
            Circular velocity at the given position(s). If the input position
            has shape ``q.shape``, the output energy will have shape
            ``q.shape[1:]``.

        """
        q = self._remove_units_prepare_shape(q)

        # Radius
        r = np.sqrt(np.sum(q**2, axis=0)) * self.units['length']
        dPhi_dxyz = self.gradient(q, t=t)
        dPhi_dr = np.sum(dPhi_dxyz * q/r.value, axis=0)

        return self.units.decompose(np.sqrt(r * np.abs(dPhi_dr)))

    # ========================================================================
    # Python special methods
    #
    def __call__(self, q):
        return self.energy(q)

    def __repr__(self):
        pars = ""
        if not isinstance(self.parameters, OrderedDict):
            keys = sorted(self.parameters.keys()) # to ensure the order is always the same
        else:
            keys = self.parameters.keys()

        for k in keys:
            v = self.parameters[k].value
            par_fmt = "{}"
            post = ""

            if hasattr(v,'unit'):
                post = " {}".format(v.unit)
                v = v.value

            if isinstance(v, float):
                if v == 0:
                    par_fmt = "{:.0f}"
                elif np.log10(v) < -2 or np.log10(v) > 5:
                    par_fmt = "{:.2e}"
                else:
                    par_fmt = "{:.2f}"

            elif isinstance(v, int) and np.log10(v) > 5:
                par_fmt = "{:.2e}"

            pars += ("{}=" + par_fmt + post).format(k,v) + ", "

        if isinstance(self.units, DimensionlessUnitSystem):
            return "<{}: {} (dimensionless)>".format(self.__class__.__name__, pars.rstrip(", "))
        else:
            return "<{}: {} ({})>".format(self.__class__.__name__, pars.rstrip(", "), ",".join(map(str, self.units._core_units)))

    def __str__(self):
        return self.__class__.__name__

    def __add__(self, other):
        if not isinstance(other, PotentialBase):
            raise TypeError('Cannot add a {} to a {}'
                            .format(self.__class__.__name__,
                                    other.__class__.__name__))

        new_pot = CompositePotential()

        if isinstance(self, CompositePotential):
            for k,v in self.items():
                new_pot[k] = v

        else:
            k = str(uuid.uuid4())
            new_pot[k] = self

        if isinstance(other, CompositePotential):
            for k,v in self.items():
                if k in new_pot:
                    raise KeyError('Potential component "{}" already exists --'
                                   'duplicate key provided in potential '
                                   'addition')
                new_pot[k] = v

        else:
            k = str(uuid.uuid4())
            new_pot[k] = other

        return new_pot

    # ========================================================================
    # Convenience methods that do fancy things
    #
    def plot_contours(self, grid, filled=True, ax=None, labels=None,
                      subplots_kw=dict(), **kwargs):
        """
        Plot equipotentials contours. Computes the potential energy on a grid
        (specified by the array `grid`).

        .. warning:: Right now the grid input must be arrays and must already
            be in the unit system of the potential. Quantity support is coming...

        Parameters
        ----------
        grid : tuple
            Coordinate grids or slice value for each dimension. Should be a
            tuple of 1D arrays or numbers.
        filled : bool (optional)
            Use :func:`~matplotlib.pyplot.contourf` instead of
            :func:`~matplotlib.pyplot.contour`. Default is ``True``.
        ax : matplotlib.Axes (optional)
        labels : iterable (optional)
            List of axis labels.
        subplots_kw : dict
            kwargs passed to matplotlib's subplots() function if an axes object
            is not specified.
        kwargs : dict
            kwargs passed to either contourf() or plot().

        Returns
        -------
        fig : `~matplotlib.Figure`

        """

        import matplotlib.pyplot as plt
        from matplotlib import cm

        # figure out which elements are iterable, which are numeric
        _grids = []
        _slices = []
        for ii,g in enumerate(grid):
            if isiterable(g):
                _grids.append((ii,g))
            else:
                _slices.append((ii,g))

        # figure out the dimensionality
        ndim = len(_grids)

        # if ndim > 2, don't know how to handle this!
        if ndim > 2:
            raise ValueError("ndim > 2: you can only make contours on a 2D grid. For other "
                             "dimensions, you have to specify values to slice.")

        if ax is None:
            # default figsize
            fig, ax = plt.subplots(1, 1, **subplots_kw)
        else:
            fig = ax.figure

        if ndim == 1:
            # 1D curve
            x1 = _grids[0][1]
            r = np.zeros((len(_grids) + len(_slices), len(x1)))
            r[_grids[0][0]] = x1

            for ii,slc in _slices:
                r[ii] = slc

            Z = self.energy(r*self.units['length']).value
            ax.plot(x1, Z, **kwargs)

            if labels is not None:
                ax.set_xlabel(labels[0])
                ax.set_ylabel("potential")
        else:
            # 2D contours
            x1,x2 = np.meshgrid(_grids[0][1], _grids[1][1])
            shp = x1.shape
            x1,x2 = x1.ravel(), x2.ravel()

            r = np.zeros((len(_grids) + len(_slices), len(x1)))
            r[_grids[0][0]] = x1
            r[_grids[1][0]] = x2

            for ii,slc in _slices:
                r[ii] = slc

            Z = self.energy(r*self.units['length']).value

            # make default colormap not suck
            cmap = kwargs.pop('cmap', cm.Blues)
            if filled:
                cs = ax.contourf(x1.reshape(shp), x2.reshape(shp), Z.reshape(shp),
                                 cmap=cmap, **kwargs)
            else:
                cs = ax.contour(x1.reshape(shp), x2.reshape(shp), Z.reshape(shp),
                                cmap=cmap, **kwargs)

            if labels is not None:
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1])

        return fig

    def plot_density_contours(self, grid, filled=True, ax=None, labels=None,
                              subplots_kw=dict(), **kwargs):
        """
        Plot density contours. Computes the density on a grid
        (specified by the array `grid`).

        .. warning:: Right now the grid input must be arrays and must already be in
            the unit system of the potential. Quantity support is coming...

        Parameters
        ----------
        grid : tuple
            Coordinate grids or slice value for each dimension. Should be a
            tuple of 1D arrays or numbers.
        filled : bool (optional)
            Use :func:`~matplotlib.pyplot.contourf` instead of
            :func:`~matplotlib.pyplot.contour`. Default is ``True``.
        ax : matplotlib.Axes (optional)
        labels : iterable (optional)
            List of axis labels.
        subplots_kw : dict
            kwargs passed to matplotlib's subplots() function if an axes object
            is not specified.
        kwargs : dict
            kwargs passed to either contourf() or plot().

        Returns
        -------
        fig : `~matplotlib.Figure`

        """

        import matplotlib.pyplot as plt
        from matplotlib import cm

        # figure out which elements are iterable, which are numeric
        _grids = []
        _slices = []
        for ii,g in enumerate(grid):
            if isiterable(g):
                _grids.append((ii,g))
            else:
                _slices.append((ii,g))

        # figure out the dimensionality
        ndim = len(_grids)

        # if ndim > 2, don't know how to handle this!
        if ndim > 2:
            raise ValueError("ndim > 2: you can only make contours on a 2D grid. For other "
                             "dimensions, you have to specify values to slice.")

        if ax is None:
            # default figsize
            fig, ax = plt.subplots(1, 1, **subplots_kw)
        else:
            fig = ax.figure

        if ndim == 1:
            # 1D curve
            x1 = _grids[0][1]
            r = np.zeros((len(_grids) + len(_slices), len(x1)))
            r[_grids[0][0]] = x1

            for ii,slc in _slices:
                r[ii] = slc

            Z = self.density(r*self.units['length']).value
            ax.plot(x1, Z, **kwargs)

            if labels is not None:
                ax.set_xlabel(labels[0])
                ax.set_ylabel("potential")
        else:
            # 2D contours
            x1,x2 = np.meshgrid(_grids[0][1], _grids[1][1])
            shp = x1.shape
            x1,x2 = x1.ravel(), x2.ravel()

            r = np.zeros((len(_grids) + len(_slices), len(x1)))
            r[_grids[0][0]] = x1
            r[_grids[1][0]] = x2

            for ii,slc in _slices:
                r[ii] = slc

            Z = self.density(r*self.units['length']).value

            # make default colormap not suck
            cmap = kwargs.pop('cmap', cm.Blues)
            if filled:
                cs = ax.contourf(x1.reshape(shp), x2.reshape(shp), Z.reshape(shp),
                                 cmap=cmap, **kwargs)
            else:
                cs = ax.contour(x1.reshape(shp), x2.reshape(shp), Z.reshape(shp),
                                cmap=cmap, **kwargs)

            # cs.cmap.set_under('w')
            # cs.cmap.set_over('k')

            if labels is not None:
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1])

        return fig

    def integrate_orbit(self, *args, **kwargs):
        """

        .. warning:: This is now deprecated. Convenient orbit integration should
            happen using the `gala.potential.Hamiltonian` class. With a
            static reference frame, you just need to pass your potential
            in to the `~gala.potential.Hamiltonian` constructor.

        Integrate an orbit in the current potential using the integrator class
        provided. Uses same time specification as `Integrator.run()` -- see
        the documentation for `gala.integrate` for more information.

        Parameters
        ----------
        w0 : `~gala.dynamics.PhaseSpacePosition`, array_like
            Initial conditions.
        Integrator : `~gala.integrate.Integrator` (optional)
            Integrator class to use.
        Integrator_kwargs : dict (optional)
            Any extra keyword argumets to pass to the integrator class
            when initializing. Only works in non-Cython mode.
        cython_if_possible : bool (optional)
            If there is a Cython version of the integrator implemented,
            and the potential object has a C instance, using Cython
            will be *much* faster.
        **time_spec
            Specification of how long to integrate. See documentation
            for `~gala.integrate.parse_time_specification`.

        Returns
        -------
        orbit : `~gala.dynamics.Orbit`

        """

        warnings.warn("Use `Hamiltonian.integrate_orbit()` instead. If you are using a "
                      "static reference frame, you just need to pass your "
                      "potential object in to the Hamiltonian constructor to use, e.g., "
                      "orbit = Hamiltonian(potential).integrate_orbit(...).",
                      DeprecationWarning)

        from ..hamiltonian import Hamiltonian
        return Hamiltonian(self).integrate_orbit(*args, **kwargs)

    def total_energy(self, x, v):
        """
        Compute the total energy (per unit mass) of a point in phase-space
        in this potential. Assumes the last axis of the input position /
        velocity is the dimension axis, e.g., for 100 points in 3-space,
        the arrays should have shape (100,3).

        Parameters
        ----------
        x : array_like, numeric
            Position.
        v : array_like, numeric
            Velocity.
        """
        warnings.warn("Use the energy methods on Orbit objects instead. In a future "
                      "release this will be removed.", DeprecationWarning)

        v = atleast_2d(v, insert_axis=1)
        return self.energy(x) + 0.5*np.sum(v**2, axis=0)

    def save(self, f):
        """
        Save the potential to a text file. See :func:`~gala.potential.save`
        for more information.

        Parameters
        ----------
        f : str, file_like
            A filename or file-like object to write the input potential object to.

        """
        from .io import save
        save(self, f)

    # ========================================================================
    # Deprecated methods
    #
    def _value(self, q, t=0.):
        warnings.warn("Use `_energy()` instead.", DeprecationWarning)
        return self._energy(q, t=t)

    def value(self, *args, **kwargs):
        __doc__ = self.energy.__doc__
        warnings.warn("Use `energy()` instead.", DeprecationWarning)
        return self.energy(*args, **kwargs)

class CompositePotential(PotentialBase, OrderedDict):
    """
    A potential composed of several distinct components. For example,
    two point masses or a galactic disk and halo, each with their own
    potential model.

    A `CompositePotential` is created like a Python dictionary, e.g.::

        >>> p1 = SomePotential(func1) # doctest: +SKIP
        >>> p2 = SomePotential(func2) # doctest: +SKIP
        >>> cp = CompositePotential(component1=p1, component2=p2) # doctest: +SKIP

    This object actually acts like an `OrderedDict`, so if you want to
    preserve the order of the potential components, use::

        >>> cp = CompositePotential() # doctest: +SKIP
        >>> cp['component1'] = p1 # doctest: +SKIP
        >>> cp['component2'] = p2 # doctest: +SKIP

    You can also use any of the built-in `Potential` classes as
    components::

        >>> from gala.potential import HernquistPotential
        >>> cp = CompositePotential()
        >>> cp['spheroid'] = HernquistPotential(m=1E11, c=10., units=(u.kpc,u.Myr,u.Msun,u.radian))

    """
    def __init__(self, *args, **kwargs):
        self._units = None
        self.ndim = None

        if len(args) > 0 and isinstance(args[0], list):
            for k,v in args[0]:
                kwargs[k] = v
        else:
            for i,v in args:
                kwargs[str(i)] = v

        self.lock = False
        for v in kwargs.values():
            self._check_component(v)

        OrderedDict.__init__(self, **kwargs)

    def __setitem__(self, key, value):
        self._check_component(value)
        super(CompositePotential, self).__setitem__(key, value)

    def _check_component(self, p):
        if not isinstance(p, PotentialBase):
            raise TypeError("Potential components may only be Potential "
                            "objects, not {0}.".format(type(p)))

        if self.units is None:
            self._units = p.units
            self.ndim = p.ndim

        else:
            if sorted([str(x) for x in self.units]) != sorted([str(x) for x in p.units]):
                raise ValueError("Unit system of new potential component must match "
                                 "unit systems of other potential components.")

            if p.ndim != self.ndim:
                raise ValueError("All potential components must have the same number of "
                                 "phase-space dimensions ({} in this case)".format(self.ndim))

        if self.lock:
            raise ValueError("Potential object is locked - new components can only be"
                             " added to unlocked potentials.")

    @property
    def units(self):  # read-only
        return self._units

    @property
    def parameters(self):
        params = dict()
        for k,v in self.items():
            params[k] = v.parameters
        return ImmutableDict(**params)

    def _energy(self, q, t=0.):
        return np.sum([p._energy(q, t) for p in self.values()], axis=0)

    def _gradient(self, q, t=0.):
        return np.sum([p._gradient(q, t) for p in self.values()], axis=0)

    def _hessian(self, w, t=0.):
        return np.sum([p._hessian(w, t) for p in self.values()], axis=0)

    def _density(self, q, t=0.):
        return np.sum([p._density(q, t) for p in self.values()], axis=0)

    def __repr__(self):
        return "<CompositePotential {}>".format(",".join(self.keys()))
