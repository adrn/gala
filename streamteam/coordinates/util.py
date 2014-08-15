# coding: utf-8

""" General utilities. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np

__all__ = ["cartesian_to_spherical", "spherical_to_cartesian"]

def cartesian_to_spherical(x, v):
    """
    Convert a position and velocity from cartesian to spherical polar
    coordinates.

    Parameters
    ----------
    x : array_like
        Position - should have shape=(n,3).
    v : array_like
        Velocity - should have shape=(n,3).
    """
    x,y,z = x.T
    vx,vy,vz = v.T
    d_xy = np.sqrt(x**2 + y**2)

    # position
    r = np.sqrt(x*x + y*y + z*z)
    phi = np.arctan2(y, x) % (2*np.pi)
    theta = np.arccos(z/r)

    # velocity
    # vr = (vx*np.cos(phi) + vy*np.sin(phi))*np.sin(theta) + vz*np.cos(theta)
    vr = (vx*x + vy*y + vz*z) / r
    vphi = (-vx*y + vy*x) / d_xy
    vphi[d_xy == 0] = 0.

    # vtheta = (vx*np.cos(phi) + vy*np.sin(phi))*np.cos(theta) - vz*np.sin(theta)
    vtheta = (vr*z/r - vz) / np.sin(theta)
    vtheta[theta == 0] = 0.

    return np.array([r,phi,theta,vr,vphi,vtheta]).T

def spherical_to_cartesian(r, phi, theta, vr, vphi, vtheta):
    """ TODO """

    x1 = r*np.sin(theta)*np.cos(phi)
    x2 = r*np.sin(theta)*np.sin(phi)
    x3 = r*np.cos(theta)

    v1 = vr*np.sin(theta)*np.cos(phi) + np.cos(theta)*np.cos(phi)*vtheta - np.sin(phi)*vphi
    v2 = vr*np.sin(theta)*np.sin(phi) + np.cos(theta)*np.sin(phi)*vtheta + np.cos(phi)*vphi
    v3 = vr*np.cos(theta) - np.sin(theta)*vtheta

    X = np.vstack((x1,x2,x3)).T
    V = np.vstack((v1,v2,v3)).T
    return X,V