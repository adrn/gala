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

__all__ = ['StaticFrame'] #, 'ConstantRotatingFrame']

cdef class CFrameWrapper:
    pass

class CFrameBase(object):

    def __init__(self, c_instance):
        self.c_instance = c_instance
