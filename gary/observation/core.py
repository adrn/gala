# coding: utf-8

""" Convenience functions for observational data """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
import astropy.units as u

__all__ = ["apparent_magnitude", "absolute_magnitude",
           "distance_modulus", "distance"]

def apparent_magnitude(mag_abs, distance):
    """
    Compute the apparent magnitude of a source given an absolute magnitude
    and a distance.

    Parameters
    ----------
    mag_abs : numeric or iterable
        Absolute magnitude.
    distance : :class:`~astropy.units.Quantity`
        The distance to the source as a Quantity object.

    Returns
    -------
    mag_app : :class:`~numpy.ndarray`
        The apparent magnitude.
    """

    # Compute the apparent magnitude
    return mag_abs + distance_modulus(distance)

def absolute_magnitude(mag_app, distance):
    """
    Compute the absolute magnitude of a source given an apparent magnitude
    and a distance.

    Parameters
    ----------
    mag_app : numeric or iterable
        Apparent magnitude.
    distance : :class:`~astropy.units.Quantity`
        The distance to the source as a Quantity object.

    Returns
    -------
    mag_abs : :class:`~numpy.ndarray`
        The absolute magnitude.
    """

    # Compute the absolute magnitude
    return mag_app - distance_modulus(distance)

def distance_modulus(distance):
    """
    Compute the distance modulus given a distance.

    Parameters
    ----------
    distance : astropy.units.Quantity
        The distance as a Quantity object.

    Returns
    -------
    distance_mod : :class:`~numpy.ndarray`
        The distance modulus.
    """

    return 5.*(np.log10(distance.to(u.pc).value) - 1.)

def distance(distance_mod):
    """
    Compute the distance from the distance modulus.

    Parameters
    ----------
    distance_mod : astropy.units.Quantity
        The distance modulus.

    Returns
    -------
    distance : :class:`~astropy.units.Quantity`
        Distance.

    """
    return 10**(distance_mod/5. + 1) * u.pc
