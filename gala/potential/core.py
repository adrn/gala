# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import OrderedDict
import warnings

# Third-party
import numpy as np
from astropy.constants import G
import astropy.units as u
from astropy.utils import isiterable, InheritDocstrings
from astropy.extern import six

# Project
from ..integrate import *
from ..util import ImmutableDict, atleast_2d
from ..units import UnitSystem, DimensionlessUnitSystem
from ..dynamics import CartesianOrbit, CartesianPhaseSpacePosition

__all__ = ["PotentialBase", "CompositePotential"]

class PotentialBase(object):
    """
    A baseclass for defining pure-Python gravitational potentials.

    Subclasses must define (at minimum) a method that evaluates
    the value (energy) of the potential at a given position ``q``
    and time ``t``: ``_value(q, t)``. For integration, the subclasses
    must also define a method to evaluate the gradient,
    ``_gradient(q,t)``. Optionally, they may also define methods
    to compute the density and hessian: ``_density()``, ``_hessian()``.
    """

    def _prefilter_pos(self, q):
        if hasattr(q, 'unit'):
            q = q.decompose(self.units).value

        q = np.ascontiguousarray(atleast_2d(q, insert_axis=1))
        return q

    def __init__(self, parameters, units=None):
        # make sure the units specified are a UnitSystem instance
        if units is not None and not isinstance(units, UnitSystem):
            units = UnitSystem(*units)

        elif units is None:
            units = DimensionlessUnitSystem()

        self.units = units

        # in case the user specified an ordered dict
        self.parameters = OrderedDict()
        for k,v in parameters.items():
            if hasattr(v, 'unit'):
                self.parameters[k] = v.decompose(self.units)
            else:
                self.parameters[k] = v*u.one

        try:
            self.G = G.decompose(self.units).value
        except u.UnitConversionError:
            self.G = 1. # TODO: this is a HACK and could lead to user confusion

    def _value(self, q, t=0.):
        raise NotImplementedError()

    def value(self, q, t=0.):
        """
        Compute the value of the potential at the given position(s).

        Parameters
        ----------
        q : `~astropy.units.Quantity`, array_like
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.

        Returns
        -------
        E : `~astropy.units.Quantity`
            The potential energy per unit mass or value of the potential.
            If the input position has shape ``q.shape``, the output energy
            will have shape ``q.shape[1:]``.
        """
        q = self._prefilter_pos(q)
        return self._value(q, t=t) * self.units['energy'] / self.units['mass']

    def _gradient(self, q, t=0.):
        raise NotImplementedError()

    def gradient(self, q, t=0.):
        """
        Compute the gradient of the potential at the given position(s).

        Parameters
        ----------
        q : `~astropy.units.Quantity`, array_like
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.

        Returns
        -------
        grad : `~astropy.units.Quantity`
            The gradient of the potential. Will have the same shape as
            the input position array, ``q``.
        """
        q = self._prefilter_pos(q)

        try:
            return self._gradient(q, t=t) * self.units['acceleration']
        except NotImplementedError:
            raise NotImplementedError("This potential has no specified gradient function.")

    def _density(self, q, t=0.):
        raise NotImplementedError()

    def density(self, q, t=0.):
        """
        Compute the density value at the given position(s).

        Parameters
        ----------
        q : `~astropy.units.Quantity`, array_like
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
        q = self._prefilter_pos(q)

        try:
            return self._density(q, t=t) * self.units['mass density']
        except NotImplementedError:
            raise NotImplementedError("This potential has no specified density function.")

    def _hessian(self, q, t=0.):
        raise NotImplementedError()

    def hessian(self, q, t=0.):
        """
        Compute the Hessian of the potential at the given position(s).

        Parameters
        ----------
        q : `~astropy.units.Quantity`, array_like
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.

        Returns
        -------
        hess : `~astropy.units.Quantity`
            TODO:
        """
        q = self._prefilter_pos(q)

        try:
            return self._hessian(q, t=t) * self.units['acceleration'] / self.units['length']
        except NotImplementedError:
            raise NotImplementedError("This potential has no specified hessian function.")

    # ========================================================================
    # Things that use the base methods
    #
    def acceleration(self, q, t=0.):
        """
        Compute the acceleration due to the potential at the given
        position(s).

        Parameters
        ----------
        q : array_like, numeric
            Position to compute the acceleration at.

        Returns
        -------
        acc : `~numpy.ndarray`
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
        x : array_like, numeric
            Position to estimate the enclossed mass.

        Returns
        -------
        menc : `~astropy.units.Quantity`
            The potential energy or value of the potential. If the input
            position has shape ``q.shape``, the output energy will have
            shape ``q.shape[1:]``.
        """
        q = self._prefilter_pos(q)

        # Fractional step-size in radius
        h = 0.01

        # Radius
        r = np.sqrt(np.sum(q**2, axis=0))

        epsilon = h*q/r[np.newaxis]

        dPhi_dr_plus = self.value(q + epsilon, t=t)
        dPhi_dr_minus = self.value(q - epsilon, t=t)
        diff = dPhi_dr_plus - dPhi_dr_minus

        if isinstance(self.units, DimensionlessUnitSystem):
            raise ValueError("No units specified when creating potential object.")
        Gee = G.decompose(self.units).value

        return np.abs(r*r * diff / Gee / (2.*h)) * self.units['mass']

    # ========================================================================
    # Python special methods
    #
    def __call__(self, q):
        return self.value(q)

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

    # ========================================================================
    # Convenience methods that do fancy things
    #
    def plot_contours(self, grid, ax=None, labels=None, subplots_kw=dict(), **kwargs):
        """
        Plot equipotentials contours. Computes the potential value on a grid
        (specified by the array `grid`).

        .. warning::

            Right now the grid input must be arrays and must already be in
            the unit system of the potential. Quantity support is coming...

        Parameters
        ----------
        grid : tuple
            Coordinate grids or slice value for each dimension. Should be a
            tuple of 1D arrays or numbers.
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

            Z = self.value(r*self.units['length']).value
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

            Z = self.value(r*self.units['length']).value

            # make default colormap not suck
            cmap = kwargs.pop('cmap', cm.Blues)
            cs = ax.contourf(x1.reshape(shp), x2.reshape(shp), Z.reshape(shp),
                             cmap=cmap, **kwargs)

            if labels is not None:
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1])

        return fig

    def plot_densty_contours(self, grid, ax=None, labels=None, subplots_kw=dict(), **kwargs):
        """
        Plot density contours. Computes the density on a grid
        (specified by the array `grid`).

        .. warning::

            Right now the grid input must be arrays and must already be in
            the unit system of the potential. Quantity support is coming...

        Parameters
        ----------
        grid : tuple
            Coordinate grids or slice value for each dimension. Should be a
            tuple of 1D arrays or numbers.
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
            r = np.zeros((len(x1), len(_grids) + len(_slices)))
            r[:,_grids[0][0]] = x1

            for ii,slc in _slices:
                r[:,ii] = slc

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

            r = np.zeros((len(x1), len(_grids) + len(_slices)))
            r[:,_grids[0][0]] = x1
            r[:,_grids[1][0]] = x2

            for ii,slc in _slices:
                r[:,ii] = slc

            Z = self.density(r*self.units['length']).value

            # make default colormap not suck
            cmap = kwargs.pop('cmap', cm.Blues)
            cs = ax.contourf(x1.reshape(shp), x2.reshape(shp), Z.reshape(shp),
                             cmap=cmap, **kwargs)
            # cs.cmap.set_under('w')
            # cs.cmap.set_over('k')

            if labels is not None:
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1])

        return fig

    def integrate_orbit(self, w0, Integrator=LeapfrogIntegrator,
                        Integrator_kwargs=dict(), cython_if_possible=True,
                        **time_spec):
        """
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
        orbit : `~gala.dynamics.CartesianOrbit`

        """

        if not isinstance(w0, CartesianPhaseSpacePosition):
            w0 = np.asarray(w0)
            ndim = w0.shape[0]//2
            w0 = CartesianPhaseSpacePosition(pos=w0[:ndim],
                                             vel=w0[ndim:])

        ndim = w0.ndim
        arr_w0 = w0.w(self.units)
        if hasattr(self, 'c_instance') and cython_if_possible:
            # WARNING TO SELF: this transpose is there because the Cython
            #   functions expect a shape: (norbits, ndim)
            arr_w0 = np.ascontiguousarray(arr_w0.T)

            # array of times
            from ..integrate.timespec import parse_time_specification
            t = np.ascontiguousarray(parse_time_specification(self.units, **time_spec))

            if Integrator == LeapfrogIntegrator:
                from ..integrate.cyintegrators import leapfrog_integrate_potential
                t,w = leapfrog_integrate_potential(self.c_instance, arr_w0, t)

            elif Integrator == DOPRI853Integrator:
                from ..integrate.cyintegrators import dop853_integrate_potential
                t,w = dop853_integrate_potential(self.c_instance, arr_w0, t,
                                                 Integrator_kwargs.get('atol', 1E-10),
                                                 Integrator_kwargs.get('rtol', 1E-10),
                                                 Integrator_kwargs.get('nmax', 0))
            else:
                raise ValueError("Cython integration not supported for '{}'".format(Integrator))

            # because shape is different from normal integrator return
            w = np.rollaxis(w, -1)
            if w.shape[-1] == 1:
                w = w[...,0]

        else:
            def acc(t, w):
                return np.vstack((w[ndim:], -self._gradient(w[:ndim], t=t)))
            integrator = Integrator(acc, func_units=self.units, **Integrator_kwargs)
            orbit = integrator.run(w0, **time_spec)
            orbit.potential = self
            return orbit

        try:
            tunit = self.units['time']
        except (TypeError, AttributeError):
            tunit = u.dimensionless_unscaled
        return CartesianOrbit.from_w(w=w, units=self.units, t=t*tunit, potential=self)

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
        return self.value(x) + 0.5*np.sum(v**2, axis=0)

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

        else:
            if sorted([str(x) for x in self.units]) != sorted([str(x) for x in p.units]):
                raise ValueError("Unit system of new potential component must match "
                                 "unit systems of other potential components.")

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

    def _value(self, q, t=0.):
        return sum([p._value(q, t) for p in self.values()])

    def _gradient(self, q, t=0.):
        return sum([p._gradient(q, t) for p in self.values()])

    def _hessian(self, w, t=0.):
        return sum([p._hessian(w, t) for p in self.values()])

    def _density(self, q, t=0.):
        return sum([p._density(q, t) for p in self.values()])

    def __repr__(self):
        return "<CompositePotential {}>".format(",".join(self.keys()))
