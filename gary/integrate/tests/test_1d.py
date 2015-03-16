# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import time
import logging

# Third-party
import numpy as np
from astropy import log as logger
from scipy.integrate import simps

# Project
from ..simpsgauss import simpson

logger.setLevel(logging.DEBUG)

plot_path = "plots/tests/TODO"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

def test_simpson():
    ncalls = 10
    func = lambda x: np.sin(x - 0.2414)*x + 2.

    x = np.linspace(0, 10, 250001)
    y = func(x)

    t0 = time.time()
    for i in range(ncalls):
        s1 = simpson(y, dx=x[1]-x[0])
    print("cython (odd): {0} sec for {1} calls".format(time.time() - t0,ncalls))

    t0 = time.time()
    for i in range(ncalls):
        s2 = simps(y, x=x)
    print("python (odd): {0} sec for {1} calls".format(time.time() - t0,ncalls))
    np.testing.assert_allclose(s1, s2)

    # -----------------------------------------------------
    print()
    x = np.linspace(0, 10, 250000)
    y = func(x)
    t0 = time.time()
    for i in range(ncalls):
        s1 = simpson(y, dx=x[1]-x[0])
    print("cython (even): {0} sec for {1} calls".format(time.time() - t0,ncalls))

    t0 = time.time()
    for i in range(ncalls):
        s2 = simps(y, x=x)
    print("python (even): {0} sec for {1} calls".format(time.time() - t0,ncalls))

    np.testing.assert_allclose(s1, s2)
