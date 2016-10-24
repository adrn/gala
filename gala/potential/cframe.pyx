# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import OrderedDict

# Third-party
from astropy.extern import six
from astropy.utils import InheritDocstrings
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython

from libc.stdio cimport printf

# Project
from .core import PotentialBase, CompositePotential
from ..util import atleast_2d
from ..units import DimensionlessUnitSystem

cdef extern from "src/funcdefs.h":
    ctypedef double (*densityfunc)(double t, double *pars, double *q) nogil
    ctypedef double (*valuefunc)(double t, double *pars, double *q) nogil
    ctypedef void (*gradientfunc)(double t, double *pars, double *q, double *grad) nogil
    ctypedef void (*hessianfunc)(double t, double *pars, double *q, double *hess) nogil

cdef extern from "src/cframe.h":
    ctypedef struct CFrame:
        valuefunc potential;
        gradientfunc gradient;
        hessianfunc hessian;

        int n_params;
        double *parameters;

__all__ = ['StaticFrame', 'ConstantRotatingFrame']

cdef class StaticFrame:
    pass

cdef class ConstantRotatingFrame:
    pass
