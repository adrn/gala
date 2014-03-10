# coding: utf-8

""" Test reading SCF files """

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np
import astropy.units as u

# Project
from ..scf import SCF
from ...units import usys

def test_simple():
    scf = SCF('/Users/adrian/projects/streams/data/simulation/Sgr/2.5e6/')
    scf = SCF('/Users/adrian/projects/streams/data/simulation/Sgr/2.5e7/')
    scf = SCF('/Users/adrian/projects/streams/data/simulation/Sgr/2.5e8/')
    scf = SCF('/Users/adrian/projects/streams/data/simulation/Sgr/2.5e9/')

def test_snap():
    scf = SCF('/Users/adrian/projects/streams/data/simulation/Sgr_DH/M2.5e+07/R0.093/4.0Gyr/L0.1')
    # scf.read_timestep('SNAP049', overwrite=True)
    d1 = scf.read_timestep('SNAP049')
    d2 = scf.read_timestep('SNAP049', units=usys)

