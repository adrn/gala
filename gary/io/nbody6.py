# coding: utf-8

""" Class for reading data from NBODY6 simulations """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
import re

# Third-party
import numpy as np
import astropy.units as u
from astropy.constants import G
from astropy.table import Table

# Project
from .core import NBodyReader

# Create logger
logger = logging.getLogger(__name__)

__all__ = ["NBODY6Reader"]

class NBODY6Reader(NBodyReader):

    def _read_units(self):
        """ """

        units = dict(length=u.pc,
                     speed=u.km/u.s,
                     dimensionless=u.dimensionless_unscaled)

        return units

    def read_snapshot(self, filename, units=None):
        """ Given a filename, read and return the data. By default,
            returns data in simulation units, but this can be changed with
            the `units` kwarg.

            Parameters
            ----------
            filename : str
                The name of the shapshot file to read.
            units : dict (optional)
                A unit system to transform the data to. If None, will return
                the data in simulation units.
        """

        # read the first line to get the numer of particles and timestep
        fullpath = os.path.join(self.path, filename)

        # column names for SNAP file, in simulation units
        colnames = "id x y z vx vy vz".split()
        coltypes = "dimensionless length length length speed speed speed".split()
        colunits = [self.sim_units[x] for x in coltypes]

        data = np.genfromtxt(fullpath, skiprows=1, names=colnames)
        if units is not None:
            new_colunits = []
            for colname,colunit in zip(colnames,colunits):
                newdata = (data[colname]*colunit).decompose(units)
                data[colname] = newdata.value
                new_colunits.append(newdata.unit)

            colunits = new_colunits

        tbl = Table(data)
        for colname,colunit in zip(colnames,colunits):
            tbl[colname].unit = colunit

        return tbl
