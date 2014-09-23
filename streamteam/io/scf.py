# coding: utf-8

""" Class for reading data from SCF simulations """

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

__all__ = ["SCFReader"]

class SCFReader(NBodyReader):

    # TODO: way to list snapshot files?

    def _read_units(self):
        """ Read and parse the SCFPAR file containing simulation parameters
            and initial conditions. Right now, only parse out the simulation
            units.
        """
        pars = dict()

        parfile = os.path.join(self.path, "SCFPAR")
        with open(parfile) as f:
            lines = f.readlines()

            # find what G is set to
            for i,line in enumerate(line):
                if line.split()[1].strip() == "G":
                    break
            pars['G'] = float(lines[i].split()[0])
            pars['length'] = float(lines[i+10].split()[0])
            pars['mass'] = float(lines[i+11].split()[0])

        _G = G.decompose(bases=[u.kpc,u.M_sun,u.Myr]).value
        X = (_G / pars['length']**3 * pars['mass'])**-0.5

        length_unit = u.Unit("{0} kpc".format(pars['length']))
        mass_unit = u.Unit("{0} M_sun".format(pars['mass']))
        time_unit = u.Unit("{:08f} Myr".format(X))

        units = dict(length=length_unit,
                     mass=mass_unit,
                     time=time_unit,
                     speed=length_unit/time_unit,
                     dimensionless=u.dimensionless_unscaled)

        return units

    def read_snap(self, filename, units=None):
        """ Given a SNAP filename, read and return the data. By default,
            returns data in simulation units, but this can be changed with
            the `units` kwarg.

            Parameters
            ----------
            filename : str
                The name of the SNAP file to read.
            units : dict (optional)
                A unit system to transform the data to. If None, will return
                the data in simulation units.
        """

        # read the first line to get the numer of particles and timestep
        fullpath = os.path.join(self.path, filename)
        with open(fullpath) as f:
            firstline = f.readline()
            try:
                nparticles,time = firstline.split()
            except:
                raise ValueError("Invalid header line. Expected 'nparticles,time', "
                                 "got:\n\t\t{}".format(firstline))
            numcols = len(f.readline().split())

        time = float(time)*self.sim_units['time']

        if numcols == 8:
            # not self gravitating
            logger.debug("Not a self-gravitating run: only 8 columns")

            # column names for SNAP file, in simulation units
            colnames = "m x y z vx vy vz s1".split()
            coltypes = "mass length length length speed speed speed dimensionless".split()
            colunits = [self.sim_units[x] for x in coltypes]

        elif numcols == 10:
            # not self gravitating
            logger.debug("A self-gravitating run: 10 columns")

            # column names for SNAP file, in simulation units
            colnames = "m x y z vx vy vz s1 s2 tub".split()
            coltypes = "mass length length length speed speed speed dimensionless dimensionless time".split()
            colunits = [self.sim_units[x] for x in coltypes]

        else:
            raise ValueError("Invalid SNAP file: {} columns (not 8 or 10).".format(numcols))

        data = np.genfromtxt(fullpath, skiprows=1, names=colnames)
        if units is not None:
            new_colunits = []
            for colname,colunit in zip(colnames,colunits):
                newdata = (data[colname]*colunit).decompose(units)
                data[colname] = newdata.value
                new_colunits.append(newdata.unit)

            time = time.decompose(units)
            colunits = new_colunits

        tbl = Table(data, meta=dict(time=time.value))
        for colname,colunit in zip(colnames,colunits):
            tbl[colname].unit = colunit

        return tbl

    def read_cen(self, units=None):
        """ Read the SCFCEN file data. By default, returns data in simulation
            units, but this can be changed with the `units` kwarg.

            Parameters
            ----------
            units : dict (optional)
                A unit system to transform the data to. If None, will return
                the data in simulation units.
        """
        fullpath = os.path.join(self.path, "SCFCEN")

        # column names for SNAP file, in simulation units
        colnames = "t dt x y z vx vy vz".split()
        coltypes = "time time length length length speed speed speed".split()
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
