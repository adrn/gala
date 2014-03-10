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

# Create logger
logger = logging.getLogger(__name__)

class SCF(object):

    def __init__(self, path):
        """ Class for reading output from an SCF simulation.

            Parameters
            ----------
            path : str
                Path to the output files, e.g., the directory containing
                SCFPAR, and SNAP files.
        """

        if not os.path.exists(path):
            raise IOError("Path to SCF output '{}' does not exist.".format(path))

        self.path = path
        self.read_scfpar()

        # figure out what timesteps are present, store metadata in a dictionary
        self.timesteps = dict()
        for filename in os.listdir(self.path):
            if not filename.startswith('SNAP') or filename.endswith('npy'):
                continue

            fullpath = os.path.join(self.path, filename)
            with open(fullpath) as f:
                nparticles,timestep = f.readline().split()

            timestep = (float(timestep)*self.sim_units['time']).to(u.Myr)
            self.timesteps[filename] = dict(nparticles=nparticles,
                                            timestep=timestep)

            if os.path.exists("{}.npy".format(fullpath)):
                self.timesteps[filename]['cache'] = os.path.split("{}.npy".format(fullpath))[1]

    def read_scfpar(self):
        """ Read and parse the SCFPAR file containing simulation parameters
            and initial conditions. Right now, only parse out the simulation
            units.
        """
        pars = dict()

        pattr = re.compile("^([0-9\.e\+]+)\s+satellite\s+((scale)|(mass))\s\(([A-Za-z]+)\)")
        parfile = os.path.join(self.path, "SCFPAR")
        with open(parfile) as f:
            for line in f:
                spl = line.split()

                # find what G is set to
                try:
                    if spl[1].strip() == 'G':
                        pars['G'] = float(spl[0])
                except IndexError:
                    pass

                # parse out length scale and mass scale
                m = pattr.search(line)
                try:
                    grp = m.groups()
                    if grp[1] == 'scale':
                        pars['length'] = float(grp[0])
                    elif grp[1] == 'mass':
                        pars['mass'] = float(grp[0])
                except:
                    pass

                # exit if have length, mass, and G
                if pars.has_key('G') and pars.has_key('length')\
                    and pars.has_key('mass'):
                    break

        _G = G.decompose(bases=[u.kpc,u.M_sun,u.Myr]).value
        X = (_G / pars['length']**3 * pars['mass'])**-0.5

        length_unit = u.Unit("{0} kpc".format(pars['length']))
        mass_unit = u.Unit("{0} M_sun".format(pars['mass']))
        time_unit = u.Unit("{:08f} Myr".format(X))

        self.sim_units = dict(length=length_unit,
                              mass=mass_unit,
                              time=time_unit,
                              speed=length_unit/time_unit)
        self.sim_units['None'] = None

    def read_timestep(self, snapfile, units=None, overwrite=False):
        """ Given a SNAP filename, read and return the data in physical
            units.

            Parameters
            ----------
            snapfile : str
                The name of the SNAP file to read. Can see all possible
                files with scf.timesteps.
            usys : dict (optional)
                A unit system to transform the data to. If None, will return
                the data in simulation units.
            overwrite : bool (optional)
                Overwrite the cached .npy file.
        """

        # numpy save file
        cache_filename = "{}.npy".format(snapfile)

        # column names for SNAP file, in simulation units
        colnames = "m x y z vx vy vz s1 s2 tub".split()
        coltypes = "mass length length length speed speed speed None None time".split()
        colunits = [self.sim_units[x] for x in coltypes]

        if not self.timesteps.has_key(snapfile):
            raise IOError("Timestep file '{}' not found!".format(snapfile))
        timestep = self.timesteps[snapfile]

        if timestep.has_key('cache') and overwrite:
            timestep.pop('cache')

        if not timestep.has_key('cache'):
            data = np.genfromtxt(os.path.join(self.path,snapfile),
                                 skiprows=1, names=colnames)
            np.save(os.path.join(self.path,cache_filename), data)
            self.timesteps[snapfile]['cache'] = cache_filename

        else:
            data = np.load(os.path.join(self.path,cache_filename))

        if units is not None:
            for colname,colunit in zip(colnames,colunits):
                if colunit is None:
                    continue
                data[colname] = (data[colname]*colunit).to(units[colunit.physical_type]).value

        return data









