# coding: utf-8

""" Base class for integrators. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Create logger
logger = logging.getLogger(__name__)

__all__ = ["Integrator"]

class Integrator(object):
    pass

class PotentialIntegrator(Integrator):
    pass