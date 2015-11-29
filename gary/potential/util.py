# coding: utf-8

""" Utilities for Potential classes """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
import astropy.units as u
import numpy as np
from sympy.utilities.lambdify import lambdify

# Project
from .core import PotentialBase

def _classnamify(s):
    s = [x.lower() for x in str(s).split()]
    words = []
    for word in s:
        words.append(word[0].upper() + word[1:])
    return "".join(words)

def from_equation(expr, vars, pars, name=None):
    """
    Create a potential class from an expression for the potential.

    Parameters
    ----------
    expr : sympy., str
    vars : iterable
        Variables, e.g., x, y, z. Should be an iterable of strings containing
        the names of the variables.
    pars : iterable
        Parameters of the potential. For example, ... TODO
    name : str (optional)
        The name of the potential class returned.
    """
    try:
        import sympy
    except ImportError:
        raise ImportError("sympy is required to use 'from_equation()' "
                          "potential class creation.")

    # convert all input to Sympy objects
    expr = sympy.sympify(expr)
    vars = [sympy.sympify(v) for v in vars]
    var_names = [v.name for v in vars]
    pars = [sympy.sympify(p) for p in pars]
    par_names = [p.name for p in pars]

    class MyPotential(PotentialBase):

        def __init__(self, units=None, **kwargs):
            self.parameters = kwargs
            for par in par_names:
                if par not in self.parameters:
                    raise ValueError("You must specify a value for "
                                     "parameter '{}'.".format(par))

            super(MyPotential,self).__init__(units)

    if name is not None:
        name = _classnamify(name)
        if "potential" not in name.lower():
            name = name + "Potential"
        MyPotential.__name__ = name

    # Energy / value
    valuefunc = lambdify(vars + pars, expr, dummify=False)
    def _value(self, w, t):
        kw = self.parameters.copy()
        for i,name in enumerate(var_names):
            kw[name] = w[i]
        return valuefunc(**kw)
    MyPotential._value = _value

    # Gradient
    gradfuncs = []
    for var in vars:
        gradfuncs.append(lambdify(vars + pars, sympy.diff(expr,var), dummify=False))

    def _gradient(self, w, t):
        kw = self.parameters.copy()
        for i,name in enumerate(var_names):
            kw[name] = w[i]
        return np.vstack([f(**kw) for f in gradfuncs])
    MyPotential._gradient = _gradient

    return MyPotential
