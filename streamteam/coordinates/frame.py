# coding: utf-8

""" Coodinate reference frames """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
from collections import defaultdict

# Third-party
import numpy as np
import astropy.units as u

# project
from .. import usys
from ..coordinates import gal_xyz_to_hel_lbd, hel_lbd_to_gal_xyz

# Create logger
logger = logging.getLogger(__name__)

_transform_graph = defaultdict(dict)
_transform_graph["galactocentric"]["heliocentric"] = gal_xyz_to_hel_lbd
_transform_graph["heliocentric"]["galactocentric"] = hel_lbd_to_gal_xyz

class ReferenceFrame(object):

    def __init__(self, name, coord_names):
        self.name = name
        self.coord_names = coord_names
        self.ndim = len(self.coord_names)

    def __repr__(self):
        return "<Frame: {}, {}D>".format(self.name, self.ndim)

    def to(self, other, X):
        """ Transform the coordinates X to the reference frame 'other'.
            X should have units of the global usys.

            Parameters
            ----------
            other : ReferenceFrame
            X : array_like
        """

        new_X = _transform_graph[self.name][other.name](X)
        return new_X

heliocentric = ReferenceFrame(name="heliocentric",
                              coord_names=("l","b","d","mul","mub","vr"))

galactocentric = ReferenceFrame(name="galactocentric",
                                coord_names=("x","y","z","vx","vy","vz"))
