# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython

# Project
from .core import Potential

__all__ = ["CPotential"]

class CPotential(Potential):
    """
    TODO:
    A baseclass for representing gravitational potentials. You must specify
    a function that evaluates the potential value (func). You may also
    optionally add a function that computes derivatives (gradient), and a
    function to compute the Hessian of the potential.

    Parameters
    ----------
    func : function
        A function that computes the value of the potential.
    gradient : function (optional)
        A function that computes the first derivatives (gradient) of the potential.
    hessian : function (optional)
        A function that computes the second derivatives (Hessian) of the potential.
    parameters : dict (optional)
        Any extra parameters that the functions (func, gradient, hessian)
        require. All functions must take the same parameters.

    """

    def __init__(self, c_class, parameters=dict()):
        # store C instance
        self.c_instance = c_class(**parameters)

        # HACK?
        super(CPotential, self).__init__(func=lambda x: x, parameters=parameters)

        # self.value = getattr(self.c_instance, 'value')
        # self.gradient = getattr(self.c_instance, 'gradient', None)
        # self.hessian = getattr(self.c_instance, 'hessian', None)

    def value(self, x):
        """
        Compute the value of the potential at the given position(s).

        Parameters
        ----------
        x : array_like, numeric
            Position to compute the value of the potential.
        """
        return self.c_instance.value(np.array(x))

    def gradient(self, x):
        """
        Compute the gradient of the potential at the given position(s).

        Parameters
        ----------
        x : array_like, numeric
            Position to compute the gradient.
        """
        try:
            return self.c_instance.gradient(np.array(x))
        except AttributeError,TypeError:
            raise ValueError("Potential C instance has no defined "
                             "gradient function")

    def hessian(self, x):
        """
        Compute the Hessian of the potential at the given position(s).

        Parameters
        ----------
        x : array_like, numeric
            Position to compute the Hessian.
        """
        try:
            return self.c_instance.hessian(np.array(x))
        except AttributeError,TypeError:
            raise ValueError("Potential C instance has no defined "
                             "Hessian function")