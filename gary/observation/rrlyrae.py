# coding: utf-8

""" Dealing with RR Lyrae photometry/measurements """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy.time import Time
import numpy as np
import astropy.units as u
try:
    from pygaia.errors.spectroscopic import vradErrorSkyAvg
    from pygaia.errors.astrometric import properMotionErrorSkyAvg, parallaxErrorSkyAvg
    PYGAIA = True
except ImportError:
    PYGAIA = False

# Project
from .core import distance_modulus
from .gaia import V_to_G

__all__ = ["M_V", "gaia_radial_velocity_error", "gaia_proper_motion_error"]

# Johnson/Cousins (V - I_C) color for RR Lyrae at *minimum*
# Guldenschuh et al. (2005 PASP 117, 721), pg. 725
# (V-I)_min = 0.579 +/- 0.006 mag
V_minus_I = 0.579

def M_V(fe_h, dfe_h=None):
    """ Given an RR Lyrae metallicity, return the V-band absolute magnitude.

        This expression comes from Benedict et al. 2011 (AJ 142, 187),
        equation 14 reads:
            M_v = (0.214 +/- 0.047)([Fe/H] + 1.5) + a_7

        where
            a_7 = 0.45 +/- 0.05

        From that, we take the absolute V-band magnitude to be:
            Mabs = 0.214 * ([Fe/H] + 1.5) + 0.45
            δMabs = sqrt[(0.047*(δ[Fe/H]))**2 + (0.05)**2]

        Parameters
        ----------
        fe_h : numeric or iterable
            Metallicity.
        dfe_h : numeric or iterable
            Uncertainty in the metallicity.

    """

    # V abs mag for RR Lyrae
    Mabs = 0.214*(fe_h + 1.5) + 0.45

    if dfe_h is not None:
        dMabs = np.sqrt((0.047*dfe_h)**2 + (0.05)**2)

        return (Mabs, dMabs)

    return Mabs

def gaia_radial_velocity_error(d, fe_h=-1.5):
    """ Compute the sky-averaged radial velocity error for an
        RR Lyrae at the given distance.

        Parameters:
        -----------
        d : quantity_like
            The distance as an Astropy Quantity object.
        fe_h : numeric
            The metallicity.
    """

    if not PYGAIA:
        raise ImportError("pygaia is required to use this function.\n"
                          "pip install pygaia")

    Vmag = distance_modulus(d) + M_V(fe_h)
    err = vradErrorSkyAvg(Vmag, spt="F0V")*u.km/u.s

    return err

def gaia_proper_motion_error(d, fe_h=-1.5):
    """ Compute the sky-averaged proper motion error for an
        RR Lyrae at the given distance.

        Parameters:
        -----------
        d : quantity_like
            The distance as an Astropy Quantity object.
        fe_h : numeric
            The metallicity.
    """

    if not PYGAIA:
        raise ImportError("pygaia is required to use this function.\n"
                          "pip install pygaia")

    Vmag = distance_modulus(d) + M_V(fe_h)
    G = V_to_G(Vmag, V_minus_I)
    err = properMotionErrorSkyAvg(G, V_minus_I)
    err = 0.5*(err[0] + err[1])
    return err*u.microarcsecond/u.yr

def gaia_parallax_error(d, fe_h=-1.5):
    """ Compute the sky-averaged parallax motion error for an
        RR Lyrae at the given distance.

        Parameters:
        -----------
        d : quantity_like
            The distance as an Astropy Quantity object.
        fe_h : numeric
            The metallicity.
    """

    if not PYGAIA:
        raise ImportError("pygaia is required to use this function.\n"
                          "pip install pygaia")

    Vmag = distance_modulus(d) + M_V(fe_h)
    G = V_to_G(Vmag, V_minus_I)
    err = parallaxErrorSkyAvg(G, V_minus_I)
    return err*u.microarcsecond

############################################################
# Below are light curve utilities
#
def sawtooth_fourier(n_max, x):
    total = np.zeros_like(x)
    for n in range(1, n_max+1):
        total += (-1)**(n+1) * 12 / (np.pi*n) * np.sin(2*np.pi*n*x)
    return -total

def time_to_phase(time, period, t0):
    """ Convert an array astropy.time.Time to an array of phases.

        Parameters
        ----------
        time : astropy.time.Time
            The grid of times to extrapolate to.
        period : astropy.units.Quantity
            Period of the source.
        t0 : astropy.time.Time
            Peak time.
    """
    return ((time.jd-t0.jd) % period.to(u.day).value) / period.to(u.day).value

def phase_to_time(phase, day, period, t0):
    """ Convert an array astropy.time.Time to an array of phases.

        Parameters
        ----------
        phase : array_like
            The grid of phases.
        day : astropy.time.Time
        period : astropy.units.Quantity
            Period of the source.
        t0 : astropy.time.Time
            Peak time.
    """

    T = period.to(u.day).value

    tt = t0.jd
    while tt < day.jd:
        tt += T
    tt -= T

    time_jd = tt + phase*T
    return Time(time_jd, format='jd', scale='utc')

def extrapolate_light_curve(time, period, t0):
    """ Extrapolate a model light curve to the given times.

        Parameters
        ----------
        time : astropy.time.Time
            The grid of times to extrapolate to.
        period : astropy.units.Quantity
            Period of the source.
        t0 : astropy.time.Time
            Peak time.
    """
    try:
        time = Time(time)
        t0 = Time(t0)
    except:
        print("You must pass in a valid astropy.time.Time object or a "
              "parseable representation for 'time' and 't0'.")
        raise

    # really simple model for an RR Lyrae light curve...
    phase_t = time_to_phase(time, period, t0)
    mag = sawtooth_fourier(25, phase_t)

    return mag
