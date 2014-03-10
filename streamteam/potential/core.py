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

__all__ = ["Potential", "CartesianPotential", "CompositePotential"]

class Potential(object):

    def __init__(self, f, f_prime=None, parameters=dict()):
        """ A baseclass for representing gravitational potentials in Cartesian
            coordinates. You must specify the functional form of the potential
            component. You may also optionally add derivatives using the
            f_prime keyword for computing accelerations.

            Parameters
            ----------
            f : function
                A function that computes the value of the potential.
            f_prime : function (optional)
                A function that computes the derivatives of the potential.
            parameters : dict (optional)
                Any extra parameters that the functions f or f_prime require.

        """

        # store parameters
        self.parameters = parameters

        # Make sure the f is callable, and that the component doesn't already
        #   exist in the potential
        if not hasattr(f, '__call__'):
            raise TypeError("'f' parameter must be a callable function! You "
                            "passed in a '{0}'".format(f.__class__))

        self.f = f

        if f_prime != None:
            if not hasattr(f_prime, '__call__'):
                raise TypeError("'f_prime' must be a callable function! You "
                                "passed in a '{0}'".format(f_prime.__class__))

        self.f_prime = f_prime

    def value_at(self, x):
        """ Compute the value of the potential at the given position(s)

            Parameters
            ----------
            x : astropy.units.Quantity, array_like, numeric
                Position to compute the value of the potential.
        """
        return self.f(x, **self.parameters)

    def acceleration_at(self, x):
        """ Compute the acceleration due to the potential at the given
            position(s)

            Parameters
            ----------
            x : astropy.units.Quantity
                Position to compute the acceleration at.
        """
        return self.f_prime(x, **self.parameters)

class CartesianPotential(Potential):

    def plot_contours(self, grid, fig=None, labels=['x','y','z'], **kwargs):
        """ Plot equipotentials contours. Takes slices at x=0, y=0, z=0,
            computes the potential value on a grid (specified by the 1D array
            `grid`). This function takes care of the meshgridding.

            Parameters
            ----------
            grid : astropy.units.Quantity
                Coordinate grid to compute the potential on. Should be a 1D
                array, and is used for all dimensions.
            fig : matplotlib.Figure (optional)
            labels : list (optional)
                A list of axis labels.
            kwargs : dict
                kwargs passed to either contourf() or plot().

        """
        figsize = kwargs.pop('figsize', (10,10))
        cmap = kwargs.pop('cmap', cm.Blues)

        if fig == None:
            fig, axes = plt.subplots(2, 2, sharex=True, sharey=True,
                                     figsize=figsize)
        else:
            axes = fig.axes

        ndim = 3
        for i in range(1,ndim):
            for jj in range(ndim-1):
                ii = i-1
                if jj > ii:
                    axes[ii,jj].set_visible(False)
                    continue

                X1, X2 = np.meshgrid(grid.value,grid.value)

                r = np.array([np.zeros_like(X1.ravel()).tolist() \
                                for xx in range(ndim)])
                r[jj] = X1.ravel()
                r[i] = X2.ravel()
                r = r.T

                Z = self.value_at(r*grid.unit).reshape(X1.shape).value
                Z = (Z - Z.min()) / (Z.max() - Z.min())

                cs = axes[ii,jj].contourf(X1, X2, Z, cmap=cmap, **kwargs)

        axes[ii,jj].set_xlim(X1.min(),X1.max())
        axes[ii,jj].set_ylim(axes[ii,jj].get_xlim())
        cax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
        fig.colorbar(cs, cax=cax)

        # Label the axes
        for jj in range(ndim-1):
            try:
                axes[-1,jj].set_xlabel("{} [{0}]".format(labels[jj], grid.unit))
            except:
                axes[-1,jj].set_xlabel("[{0}]".format(grid.unit))

            try:
                axes[jj,0].set_ylabel("{} [{0}]".format(labels[jj+1], grid.unit))
            except:
                axes[jj,0].set_ylabel("[{0}]".format(grid.unit))

        fig.subplots_adjust(hspace=0.1, wspace=0.1, left=0.08, bottom=0.08, top=0.9, right=0.9 )

        return fig, axes

class CompositePotential(dict, CartesianPotential):

    def __init__(self, **kwargs):
        """ Represents a potential composed of several distinct potential
            components. For example, two point masses or a galactic disk
            + halo.

            Parameters
            ----------
            kwargs

        """
        if len(kwargs) == 0:
            raise ValueError("You must specify at least one potential "
                             "component!")

        for v in kwargs.values():
            if not isinstance(v, Potential):
                raise TypeError("Values may only be Potential objects, not "
                                "{0}.".format(type(v)))

        dict.__init__(self, **kwargs)

    def __repr__(self):
        return "<CompositePotential: {0}>".format(", ".join(self.keys()))

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
        for p in self.values():
            try:
                v = v + p.value_at(x)
            except NameError:
                v = p.value_at(x)
        return v

    def acceleration_at(self, x):
        """ Compute the acceleration due to the potential at the given
            position(s)

            Parameters
            ----------
            x : astropy.units.Quantity
                Position to compute the acceleration at.
        """
        for p in self.values():
            try:
                v = v + p.acceleration_at(x)
            except NameError:
                v = p.acceleration_at(x)
        return v