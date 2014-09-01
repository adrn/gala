#!/usr/bin/env python
# coding: utf-8

""" Plot a given snap file. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from astropy import log as logger
import astropy.units as u

# Project
import streamteam.dynamics as sd
import streamteam.io as sio
from streamteam.units import galactic

def main(snap_path):
    path,snap_file = os.path.split(snap_path)

    scf = sio.SCFReader(path)
    tbl = scf.read_snap(snap_file, units=galactic)
    w = sio.tbl_to_w

    fig = sd.plot_orbits(w, marker='.', linestyle='none', alpha=0.1)
    plt.show()

if __name__ == '__main__':
    main(str(sys.argv[1]))