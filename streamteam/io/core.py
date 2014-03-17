# coding: utf-8

""" Base class readers """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import abc
import os, sys
import logging

# Third-party
import numpy as np

# Create logger
logger = logging.getLogger(__name__)

class NBodyReader(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _read_units(self):
        """ Read a parameter file and return a dictionary of simulation
            units as astropy.units.Unit objects.
        """
        return

    # @abc.abstractmethod
    # def save(self, output, data):
    #     """Save the data object to the output."""
    #     return

    def __init__(self, path):
        """ Class for reading output from an n-body simulation.

            Parameters
            ----------
            path : str
                Path to the output files.
        """

        if not os.path.exists(path):
            raise IOError("Path to output '{}' does not exist.".format(path))

        self.path = path
        self.sim_units = self._read_units()
        self.nparticles = None