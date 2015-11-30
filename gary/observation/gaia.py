# coding: utf-8

""" Observe stars with Gaia-like uncertainties. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np
import astropy.units as u

__all__ = []

def V_to_G(V, V_minus_I):
    """ Convert Johnson V to Gaia G-band.

        Parameters
        ----------
        V : numeric or iterable
            The V-band apparent magnitude of a source.
        V_minus_I : numeric or iterable
            The V-I color of the source.
    """
    return V - 0.0257 - 0.0924*V_minus_I - 0.1623*V_minus_I**2 + 0.009*V_minus_I**3
