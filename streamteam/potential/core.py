# coding: utf-8

""" Base class for handling analytic representations of scalar gravitational
    potentials.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import copy
import inspect
import logging
import functools

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import astropy.units as u
from astropy.utils import isiterable

__all__ = ["Potential", "CompositePotential"]

class Potential(object):

    def __init__(self, func, gradient=None, hessian=None, parameters=dict()):
        """ A baseclass for representing gravitational potentials. You must specify
            a function that evaluates the potential value (func). You may also optionally
            add a function that computes derivatives (gradient), and a function to compute
            the Hessian of the potential.

            Parameters
            ----------
            func : function
                A function that computes the value of the potential.
            gradient : function (optional)
                A function that computes the first derivatives (gradient) of the potential.
            hessian : function (optional)
                A function that computes the second derivatives (Hessian) of the potential.
            parameters : dict (optional)
                Any extra parameters that the functions f require. All functions must take
                the same parameters.

        """

        # store parameters
        self.parameters = parameters

        # Make sure the functions are callable
        for f in [func, gradient, hessian]:
            if f is not None and not hasattr(f, '__call__'):
                raise TypeError("'{}' parameter must be callable! You passed "
                                "in a '{}'".format(f.func_name, f.__class__))

        self.func = func
        self.gradient = gradient
        self.hessian = hessian

    def value_at(self, x):
        """ Compute the value of the potential at the given position(s)

            Parameters
            ----------
            x : array_like, numeric
                Position to compute the value of the potential.
        """
        return self.func(x, **self.parameters)

    def acceleration_at(self, x):
        """ Compute the acceleration due to the potential at the given
            position(s).

            Parameters
            ----------
            x : array_like, numeric
                Position to compute the acceleration at.
        """
        return -self.gradient(x, **self.parameters)

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

    def plot_contours(self, grid, ax=None, labels=None, subplots_kw=dict(), **kwargs):
        """ Plot equipotentials contours. Computes the potential value on a grid
            (specified by the array `grid`).

            Parameters
            ----------
            grid : tuple
                Coordinate grids or slice value for each dimension. Should be a
                tuple of 1D array (or Quantity) objects.
            ax : matplotlib.Axes (optional)
            labels : iterable (optional)
                List of axis labels.
            subplots_kw : dict
                kwargs passed to matplotlib's subplots() function if an axes object
                is not specified.
            kwargs : dict
                kwargs passed to either contourf() or plot().

        """

        # figure out which elements are iterable, which are numeric
        _grids = []
        _slices = []
        for ii,g in enumerate(grid):
            if not hasattr(g,'unit'):
                g = g*u.dimensionless_unscaled

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

        # use the unit from the first grid
        _unit = _grids[0][1].unit
        if labels is not None:
            labels = ["{} [{}]".format(l,_unit) for l in labels]

        if ndim == 1:
            # 1D curve
            x1 = _grids[0][1].value
            r = np.zeros((len(x1), len(_grids) + len(_slices)))
            r[:,_grids[0][0]] = x1

            for ii,slc in _slices:
                r[:,ii] = slc.to(_unit).value

            Z = self.value_at(r*_unit)
            ax.plot(x1, Z.value, **kwargs)

            if labels is not None:
                ax.set_xlabel(labels[0])

                if Z.unit is not u.dimensionless_unscaled:
                    ax.set_ylabel("potential value [{}]".format(Z.unit))
                else:
                    ax.set_ylabel("potential value")
        else:
            # 2D contours
            x1,x2 = np.meshgrid(_grids[0][1].to(_unit).value,
                                _grids[1][1].to(_unit).value)
            shp = x1.shape
            x1,x2 = x1.ravel(), x2.ravel()

            r = np.zeros((len(x1), len(_grids) + len(_slices)))
            r[:,_grids[0][0]] = x1
            r[:,_grids[1][0]] = x2

            for ii,slc in _slices:
                r[:,ii] = slc.to(_unit).value

            Z = self.value_at(r*_unit).value

            # make default colormap not suck
            cmap = kwargs.pop('cmap', cm.Blues)
            cs = ax.contourf(x1.reshape(shp), x2.reshape(shp), Z.reshape(shp),
                             cmap=cmap, **kwargs)

            if labels is not None:
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1])

        return fig,ax

class CompositePotential(dict, Potential):

    def __init__(self, **kwargs):
        """ Represents a potential composed of several distinct potential
            components. For example, two point masses or a galactic disk
            + halo.

            Parameters
            ----------
            kwargs

        """

        for v in kwargs.values():
            if not isinstance(v, Potential):
                raise TypeError("Values may only be Potential objects, not "
                                "{0}.".format(type(v)))

        dict.__init__(self, **kwargs)

    def __setitem__(self, key, value):
        if not isinstance(value, Potential):
            raise TypeError("Values may only be Potential objects, not "
                            "{0}.".format(type(value)))

        super(CompositePotential, self).__setitem__(key, value)

    def value_at(self, x):
        """ Compute the value of the potential at the given position(s)

            Parameters
            ----------
            x : astropy.units.Quantity, array_like, numeric
                Position to compute the value of the potential.
        """
        return u.Quantity([p.value_at(x) for p in self.values()]).sum(axis=0)

    def acceleration_at(self, x):
        """ Compute the acceleration due to the potential at the given
            position(s)

            Parameters
            ----------
            x : astropy.units.Quantity
                Position to compute the acceleration at.
        """
        return u.Quantity([p.acceleration_at(x) for p in self.values()]).sum(axis=0)