# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import OrderedDict

# Third-party
import numpy as np
from astropy.constants import G
import astropy.units as u
from astropy.utils import isiterable

# Project
from ..integrate import *
from ..util import inherit_docs, ImmutableDict, atleast_2d
from ..units import UnitSystem
from ..dynamics import CartesianOrbit, CartesianPhaseSpacePosition

__all__ = ["PotentialBase", "CompositePotential"]

class PotentialBase(object):
    """
    A baseclass for defining gravitational potentials.

    Subclasses must define a function that evaluates the value of the
    potential at a given position and time. For integration, the
    subclasses should also define a gradient function. Optionally, they
    may also define functions to compute the density and hessian.
    """
    def __init__(self, units=None):
        # make sure the units specified are a UnitSystem instance
        if units is not None and not isinstance(units, UnitSystem):
            units = UnitSystem(*units)
        self.units = units

        # must set parameters first...
        if not hasattr(self, 'parameters'):
            raise ValueError("Must set parameters of potential subclass before"
                             " calling super().")

        for k,v in self.parameters.items():
            if hasattr(v, 'unit'):
                self.parameters[k] = v.decompose(self.units).value

    def _value(self):
        raise NotImplementedError()

    def value(self, q, t=0.):
        """
        Compute the value of the potential at the given position(s).

        Parameters
        ----------
        q : array_like, numeric
            Position to compute the value of the potential.

        Returns
        -------
        E : `~numpy.ndarray`
            The potential energy, value of the potential. Will have
            the same shape as the input position, array, ``q``, but
            without the coordinate axis, ``axis=0``.
        """
        q = np.ascontiguousarray(atleast_2d(q, insert_axis=1))
        return self._value(q, t=t)

    def _gradient(self, *args, **kwargs):
        raise NotImplementedError()

    def gradient(self, q, t=0.):
        """
        Compute the gradient of the potential at the given position(s).

        Parameters
        ----------
        q : array_like, numeric
            Position to compute the gradient.

        Returns
        -------
        grad : `~numpy.ndarray`
            The gradient of the potential. Will have the same shape as
            the input position array, ``q``.
        """
        q = np.ascontiguousarray(atleast_2d(q, insert_axis=1))
        try:
            return self._gradient(q, t=t)
        except NotImplementedError:
            raise NotImplementedError("This potential has no specified gradient function.")

    def _density(self, *args, **kwargs):
        raise NotImplementedError()

    def density(self, q, t=0.):
        """
        Compute the density value at the given position(s).

        Parameters
        ----------
        q : array_like, numeric
            Position to compute the density.

        Returns
        -------
        dens : `~numpy.ndarray`
            The density. Will have the same shape as the input position,
            array, ``q``, but without the coordinate axis, ``axis=0``.
        """
        q = np.ascontiguousarray(atleast_2d(q, insert_axis=1))
        try:
            return self._density(q, t=t)
        except NotImplementedError:
            raise NotImplementedError("This potential has no specified density function.")

    def _hessian(self, *args, **kwargs):
        raise NotImplementedError()

    def hessian(self, q, t=0.):
        """
        Compute the Hessian of the potential at the given position(s).

        Parameters
        ----------
        q : array_like, numeric
            Position to compute the Hessian.
        """
        q = np.ascontiguousarray(atleast_2d(q, insert_axis=1))
        try:
            return self._hessian(q, t=t)
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
        acce : `~numpy.ndarray`
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
        menc : `~numpy.ndarray`
            The mass. Will have the same shape as the input position,
            array, ``q``, but without the coordinate axis, ``axis=0``
        """

        q = np.ascontiguousarray(atleast_2d(q, insert_axis=1))

        # Fractional step-size in radius
        h = 0.01

        # Radius
        r = np.sqrt(np.sum(q**2, axis=0))

        epsilon = h*q/r[np.newaxis]

        dPhi_dr_plus = self.value(q + epsilon, t=t)
        dPhi_dr_minus = self.value(q - epsilon, t=t)
        diff = dPhi_dr_plus - dPhi_dr_minus

        if self.units is None:
            raise ValueError("No units specified when creating potential object.")
        Gee = G.decompose(self.units).value

        return np.abs(r*r * diff / Gee / (2.*h))

    # ========================================================================
    # Python special methods
    #
    def __call__(self, q):
        return self.value(q)

    def __repr__(self):
        pars = ""
        for k,v in self.parameters.items():
            par_fmt = "{}"
            post = ""

            if hasattr(v,'unit'):
                post = " {}".format(v.unit)
                v = v.value

            if isinstance(v, float):
                if np.log10(v) < -2 or np.log10(v) > 5:
                    par_fmt = "{:.2e}"
                else:
                    par_fmt = "{:.2f}"

            elif isinstance(v, int) and np.log10(v) > 5:
                par_fmt = "{:.2e}"

            pars += ("{}=" + par_fmt + post).format(k,v) + ", "

        if self.units is None:
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

            Z = self.value(r)
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

            Z = self.value(r)

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

            Z = self.density(r)
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

            Z = self.density(r)

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
        the documentation for `gary.integrate` for more information.

        Parameters
        ----------
        w0 : `~gary.dynamics.PhaseSpacePosition`, array_like
            Initial conditions.
        Integrator : `~gary.integrate.Integrator` (optional)
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
            for `~gary.integrate.parse_time_specification`.

        Returns
        -------
        orbit : `~gary.dynamics.CartesianOrbit`

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
            t = parse_time_specification(**time_spec)

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
            acc = lambda t,w: np.vstack((w[ndim:], self.acceleration(w[:ndim], t=t)))
            integrator = Integrator(acc, **Integrator_kwargs)
            t,w = integrator.run(arr_w0, **time_spec)

        return CartesianOrbit.from_w(w=w, units=self.units, t=t, potential=self)

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
        # TODO: deprecationwarning?
        v = atleast_2d(v, insert_axis=1)
        return self.value(x) + 0.5*np.sum(v**2, axis=0)

    def save(self, f):
        """
        Save the potential to a text file. See :func:`~gary.potential.save`
        for more information.

        Parameters
        ----------
        f : str, file_like
            A filename or file-like object to write the input potential object to.

        """
        from .io import save
        save(self, f)

@inherit_docs
class CompositePotential(PotentialBase, OrderedDict):
    """
    A potential composed of several distinct components. For example,
    two point masses or a galactic disk and halo, each with their own
    potential model.

    TODO: needs re-writing

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

        >>> from gary.potential import HernquistPotential
        >>> cp = CompositePotential()
        >>> cp['spheroid'] = HernquistPotential(m=1E11, c=10., units=(u.kpc,u.Myr,u.Msun,u.radian))

    """
    def __init__(self, **kwargs):
        self._units = None

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

    @property
    def units(self):  # read-only
        return self._units

    @property
    def parameters(self):
        params = dict()
        for k,v in self.items():
            params[k] = v.parameters
        return ImmutableDict(params)

    def value(self, q, t=0.):
        return np.array([p.value(q, t) for p in self.values()]).sum(axis=0)

    def gradient(self, q, t=0.):
        return np.array([p.gradient(q, t) for p in self.values()]).sum(axis=0)

    def hessian(self, w, t=0.):
        return np.array([p.hessian(w, t) for p in self.values()]).sum(axis=0)

    def density(self, q, t=0.):
        return np.array([p.density(q, t) for p in self.values()]).sum(axis=0)
