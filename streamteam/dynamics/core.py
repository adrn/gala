# coding: utf-8

""" General dynamics utilities. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from astropy import log as logger
import astropy.units as u

# Project

__all__ = ['angular_momentum']

def angular_momentum(w):
    """
    Compute the angular momentum vector(s) of phase-space point(s), `w`.

    Parameters
    ----------
    w : array_like
        Array of phase-space positions. The last axis (`axis=-1`) is assumed
        to be the phase-space dimension so that the phase-space dimensionality
        is `w.shape[-1]`.
    """
    ndim = w.shape[-1]
    return np.cross(w[...,:ndim//2], w[...,ndim//2:])