# coding: utf-8

""" Convenience functions for observational data """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u

__all__ = ["apparent_magnitude", "absolute_magnitude", "distance_modulus",
           "distance"]

def apparent_magnitude(M, d):
    """ Compute the apparent magnitude of a source given an absolute magnitude
        and a distance.

        Parameters
        ----------
        M : numeric or iterable
            Absolute magnitude.
        d : astropy.units.Quantity
            The distance to the source as a Quantity object.

    """

    if not isinstance(d, u.Quantity):
        raise TypeError("Distance must be an Astropy Quantity object!")

    # Compute the apparent magnitude -- ignores extinction
    return M + distance_modulus(d)

def absolute_magnitude(m, d):
    """ Compute the absolute magnitude of a source given an apparent magnitude
        and a distance.

        Parameters
        ----------
        m : numeric or iterable
            Apparent magnitude of a source.
        d : astropy.units.Quantity
            The distance to the source as a Quantity object.
    """

    if not isinstance(d, u.Quantity):
        raise TypeError("Distance must be an Astropy Quantity object!")

    # Compute the apparent magnitude -- ignores extinction
    return m - distance_modulus(d)

def distance_modulus(d):
    """ Compute the distance modulus given a distance.

        Parameters
        ----------
        d : astropy.units.Quantity
            The distance as a Quantity object.
    """
    if not isinstance(d, u.Quantity):
        raise TypeError("Distance must be an Astropy Quantity object!")

    return 5.*(np.log10(d.to(u.pc).value) - 1.)

def distance(distance_modulus):
    """ Compute the distance from the distance modulus.

        Parameters:
        dm : numeric or iterable
            The distance modulus
    """
    return 10**(distance_modulus/5. + 1) * u.pc
