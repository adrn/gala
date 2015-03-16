# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
import numpy as np

__all__ = ['simpson']

cdef extern from "1d/simpson.h":
    double _simpson (double *y, double dx, int n)

cpdef simpson(double[::1] y, double dx):
    return _simpson(&y[0], dx, y.size)
