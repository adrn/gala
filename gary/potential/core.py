# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import abc

# Third-party
import numpy as np
from astropy.constants import G
import astropy.units as u
from astropy.utils import isiterable
import six

# Project
from ..integrate import *
from ..util import inherit_docs, ImmutableDict

__all__ = ["PotentialBase", "CompositePotential"]

class MetaPotential(type):
    # def __init__(cls, name, bases, dct):
    #     super(MetaPotential, cls).__init__(name, bases, dct)

    #     if name != 'PotentialBase' and 'units' not in dct:
    #         raise NotImplementedError("Potentials must have a 'units' attribute that "
    #                                   "defines a valid system of units.")

    #     if name != 'PotentialBase' and 'parameters' not in dct:
    #         raise NotImplementedError("Potentials must have a 'parameters' attribute that "
    #                                   "contains the parameters to pass to its methods.")
    pass


@six.add_metaclass(MetaPotential)
class PotentialBase(object):
    """
    A baseclass for defining gravitational potentials.

    You must specify
    a function that evaluates the potential value (func). You may also
    optionally add a function that computes derivatives (gradient), and a
    function to compute the Hessian of the potential.

    Parameters
    ----------
    func : function
        A function that computes the value of the potential.
    units : iterable
        A list of astropy.units.Unit objects that define a complete unit system.
        Must include at least a length unit, time unit, and mass unit.
    gradient : function (optional)
        A function that computes the first derivatives (gradient) of the potential.
    hessian : function (optional)
        A function that computes the second derivatives (Hessian) of the potential.
    parameters : dict (optional)
        Any extra parameters that the functions (func, gradient, hessian)
        require. All functions must take the same parameters.

    """

    def __init__(self, units=None, **parameters):
        self.units = units
        self.parameters = parameters

    @abc.abstractmethod
    def _value(self):
        raise NotImplementedError()

    def value(self, x):
        """
        Compute the value of the potential at the given position(s).

        Parameters
        ----------
        x : array_like, numeric
            Position to compute the value of the potential.
        """
        return self._value(np.atleast_2d(x), **self.parameters)

    def _gradient(self, *args, **kwargs):
        raise NotImplementedError()

    def gradient(self, x):
        """
        Compute the gradient of the potential at the given position(s).

        Parameters
        ----------
        x : array_like, numeric
            Position to compute the gradient.
        """
        if self._gradient is None:
            raise NotImplementedError("No gradient function was specified when"
                                      " the object was created!")
        return self._gradient(np.atleast_2d(x), **self.parameters)

    def _hessian(self, *args, **kwargs):
        raise NotImplementedError()

    def hessian(self, x):
        """
        Compute the Hessian of the potential at the given position(s).

        Parameters
        ----------
        x : array_like, numeric
            Position to compute the Hessian.
        """
        if self._hessian is None:
            raise NotImplementedError("No Hessian function was specified when"
                                      " the object was created!")
        return self._hessian(np.atleast_2d(x), **self.parameters)

    # ========================================================================
    # Things that use the base methods
    #
    def acceleration(self, x):
        """
        Compute the acceleration due to the potential at the given
        position(s).

        Parameters
        ----------
        x : array_like, numeric
            Position to compute the acceleration at.
        """
        return -self.gradient(x)

    def mass_enclosed(self, x):
        """
        Estimate the mass enclosed within the given position by assumine the potential
        is spherical. This is basic, and assumes spherical symmetry.

        Parameters
        ----------
        x : array_like, numeric
            Position to estimate the enclossed mass.
        """

        # Fractional step-size in radius
        h = 0.01

        # Radius
        r = np.sqrt(np.sum(x**2, axis=-1))

        epsilon = h*x/r[...,np.newaxis]

        dPhi_dr_plus = self.value(x + epsilon)
        dPhi_dr_minus = self.value(x - epsilon)
        diff = dPhi_dr_plus - dPhi_dr_minus

        if self.units is None:
            raise ValueError("No units specified when creating potential object.")
        Gee = G.decompose(self.units).value

        return np.abs(r*r * diff / Gee / (2.*h))

    # ========================================================================
    # Python special methods
    #
    def __call__(self, x):
        return self.value(x)

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

        return "<{}: {}>".format(self.__class__.__name__, pars.rstrip(", "))

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

            r = np.zeros((len(x1), len(_grids) + len(_slices)))
            r[:,_grids[0][0]] = x1
            r[:,_grids[1][0]] = x2

            for ii,slc in _slices:
                r[:,ii] = slc

            Z = self.value(r)

            # make default colormap not suck
            cmap = kwargs.pop('cmap', cm.Blues)
            cs = ax.contourf(x1.reshape(shp), x2.reshape(shp), Z.reshape(shp),
                             cmap=cmap, **kwargs)

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
        w0 : array_like
            Initial conditions.
        Integrator : class
            Integrator class to use.

        Other Parameters
        ----------------
        (see Integrator documentation)

        """

        if Integrator == LeapfrogIntegrator:
            if hasattr(self, 'c_instance') and cython_if_possible:
                from ..integrate._leapfrog import cy_leapfrog_run
                from ..integrate.timespec import _parse_time_specification

                # use fast integrator
                times = _parse_time_specification(**time_spec)
                nsteps = len(times) - 1
                dt = times[1] - times[0]
                t1 = times[0]

                w0 = np.ascontiguousarray(np.atleast_2d(w0))
                return cy_leapfrog_run(self.c_instance, w0, dt, nsteps, t1)

            else:
                acc = lambda t,w: self.acceleration(w)
        else:
            acc = lambda t,w: np.hstack((w[...,3:],self.acceleration(w[...,:3])))

        integrator = Integrator(acc, **Integrator_kwargs)
        return integrator.run(w0, **time_spec)

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
        return self.value(x) + 0.5*np.sum(v**2, axis=-1)

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
class CompositePotential(PotentialBase, dict):
    """
    A potential composed of several distinct components. For example,
    two point masses or a galactic disk and halo, each with their own
    potential model.

    TODO: needs re-writing

    A `CompositePotential` is created like a Python dictionary, e.g.::

        >>> p1 = Potential(func1)
        >>> p2 = Potential(func2)
        >>> cp = CompositePotential(component1=p1, component2=p2)

    or equivalently::

        >>> cp = CompositePotential()
        >>> cp['component1'] = p1
        >>> cp['component2'] = p2

    You can also use any of the built-in `Potential` classes as
    components::

        >>> from gary.potential import HernquistPotential
        >>> cp = CompositePotential()
        >>> cp['spheroid'] = HernquistPotential(m=1E11, c=10., units=(u.kpc,u.Myr,u.Msun))

    """
    def __init__(self, **kwargs):
        self._units = None

        for v in kwargs.values():
            self._check_component(v)

        dict.__init__(self, **kwargs)

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

    def value(self, x):
        x = np.atleast_2d(x).copy()
        return np.array([p.value(x) for p in self.values()]).sum(axis=0)

    def gradient(self, x):
        x = np.atleast_2d(x).copy()
        return np.array([p.gradient(x) for p in self.values()]).sum(axis=0)

    def hessian(self, x):
        x = np.atleast_2d(x).copy()
        return np.array([p.hessian(x) for p in self.values()]).sum(axis=0)
