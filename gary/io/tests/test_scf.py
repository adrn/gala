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
from ..scf import SCFReader
from ...units import galactic

def test_simple():
    scf = SCFReader('/Users/adrian/projects/gary/tests/M2.5e+08')

def test_snap():
    scf = SCFReader('/Users/adrian/projects/gary/tests/M2.5e+08')
    d1 = scf.read_snap('SNAP113')
    print(d1.meta)

    d1 = scf.read_snap('SNAP113', units=galactic)
    print(d1.meta)
