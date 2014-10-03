# coding: utf-8

""" Test dynamics plotting  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import matplotlib.pyplot as plt
import numpy as np
from astropy import log as logger

# Project
from ..plot import *

logger.setLevel(logging.DEBUG)

plot_path = "plots/tests/dynamics/plot"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

def test_orbits():

    # generate an "orbit"
    n = 8
    t = np.linspace(0, 100, 1000).reshape(1000,1)
    x = np.cos(np.random.uniform(1.,8.,size=(1,n))*t).T[None]
    y = np.cos(np.random.uniform(1.,8.,size=(1,n))*t).T[None]
    z = np.cos(np.random.uniform(1.,8.,size=(1,n))*t).T[None]

    w = np.vstack((x,y,z)).T

    fig = plot_orbits(w, linestyle='none', marker='.', alpha=0.25)
    fig.savefig(os.path.join(plot_path, "all_orbits.png"))

    fig = plot_orbits(w, ix=0, linestyle='none', marker='.', alpha=0.25)
    fig.savefig(os.path.join(plot_path, "one_orbit.png"))

    fig = plot_orbits(w, triangle=True, linestyle='-', marker=None)
    fig.savefig(os.path.join(plot_path, "all_orbits_triangle.png"))

    fig = plot_orbits(w, ix=0, triangle=True, linestyle='-', marker=None)
    fig.savefig(os.path.join(plot_path, "one_orbit_triangle.png"))

def test_three_panel():

    q = np.random.uniform(0.,10.,size=(1000,3))

    fig = three_panel(q)
    fig.savefig(os.path.join(plot_path, "three-panel-random.png"))

    fig = three_panel(q, triangle=True)
    fig.savefig(os.path.join(plot_path, "three-panel-random_triangle.png"))
