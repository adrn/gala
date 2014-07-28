# coding: utf-8

""" Common astronomical potentials. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import uuid
import math

# Third-party
import numpy as np
import astropy.units as u
from astropy.constants import G

from .core import Potential, CompositePotential

__all__ = ["PointMassPotential", "MiyamotoNagaiPotential",\
           "HernquistPotential", "TriaxialLogarithmicPotential",\
           "IsochronePotential"]

############################################################
#    Potential due to a point mass at a given position
#
def point_mass_funcs(units):
    # scale G to be in this unit system
    if units is None:
        _G = 1.
    else:
        _G = G.decompose(units).value

    def f(x, x0, m):
        xx = x-x0
        R = np.sqrt(np.sum(xx**2, axis=-1))
        return -_G * m / R

    def gradient(x, x0, m):
        xx = x-x0
        a = np.sum(xx**2, axis=-1)**-1.5
        return _G * m * xx * a

    def hessian(x, x0, m):
        raise NotImplementedError() # TODO:

    return f, gradient, None

class PointMassPotential(Potential):
    r"""
    Represents a point-mass potential at the given origin.

    .. math::

        \Phi = -\frac{Gm}{x-x0}

    Parameters
    ----------
    m : numeric
        Mass.
    x0 : array_like, numeric
        Position of the point mass relative to origin of coordinates
    usys : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.
    """

    def __init__(self, m, x0, usys=None):
        parameters = dict(m=m, x0=x0)
        func,gradient,hessian = point_mass_funcs(usys)
        super(PointMassPotential, self).__init__(func=func, gradient=gradient,
                                                 hessian=hessian,
                                                 parameters=parameters)

##############################################################################
#    Isochrone potential
#

def isochrone_funcs(units):
    # scale G to be in this unit system
    if units is None:
        _G = 1.
    else:
        _G = G.decompose(units).value

    def func(x, m, b):
        r2 = np.sum(x**2, axis=-1)
        val = -_G * m / (np.sqrt(r2 + b*b) + b)
        return val

    def gradient(x, m, b):
        r2 = np.sum(x**2, axis=-1)
        fac = _G*m / (np.sqrt(r2 + b*b) + b)**2 / np.sqrt(r2 + b*b)
        return fac[...,None] * x

    def hessian(x, m, b):
        raise NotImplementedError() # TODO:

    return func, gradient, None

class IsochronePotential(Potential):
    r"""
    The Isochrone potential.

    .. math::

        \Phi_{spher} = -\frac{GM}{\sqrt{r^2+b^2} + b}

    Parameters
    ----------
    m : numeric
        Mass.
    b : numeric
        Core concentration.
    usys : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, b, usys=None):
        parameters = dict(m=m, b=b)
        func,gradient,hessian = isochrone_funcs(usys)
        super(IsochronePotential, self).__init__(func=func, gradient=gradient,
                                                 hessian=hessian,
                                                 parameters=parameters)

##############################################################################
#    Miyamoto-Nagai Disk potential from Miyamoto & Nagai 1975
#    http://adsabs.harvard.edu/abs/1975PASJ...27..533M
#
def miyamoto_nagai_funcs(units):
    # scale G to be in this unit system
    if units is None:
        _G = 1.
    else:
        _G = G.decompose(units).value

    def func(xyz, m, a, b):
        x,y,z = xyz.T
        z_term = a + np.sqrt(z*z + b*b)
        return -_G * m / np.sqrt(x*x + y*y + z_term*z_term)

    def gradient(xyz, m, a, b):
        x,y,z = xyz.T

        sqrtz = np.sqrt(z*z + b*b)
        z_term = a + sqrtz
        fac = _G*m*(x*x + y*y + z_term*z_term)**-1.5

        dx = fac*x
        dy = fac*y

        c = a / sqrtz
        dz = fac*z * (1. + c)

        return np.array([dx,dy,dz]).T

    def hessian(xyz, m, a, b): # TODO
        pass

    return func, gradient, None

class MiyamotoNagaiPotential(Potential):
    r"""
    Miyamoto-Nagai potential (1975) for a disk-like potential.

    .. math::

        \Phi_{disk} = -\frac{GM_{disk}}{\sqrt{R^2 + (a + sqrt{z^2 + b^2})^2}}

    Parameters
    ----------
    m : numeric
        Mass.
    a : numeric
    b : numeric
    usys : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, a, b, usys=None):
        parameters = dict(m=m, a=a, b=b)
        func,gradient,hessian = miyamoto_nagai_funcs(usys)
        super(MiyamotoNagaiPotential, self).__init__(func=func,
                                                     gradient=gradient,
                                                     hessian=hessian,
                                                     parameters=parameters)

# HERE

##############################################################################
#    Hernquist Spheroid potential from Hernquist 1990
#    http://adsabs.harvard.edu/abs/1990ApJ...356..359H
#
def hernquist_funcs(units):
    # scale G to be in this unit system
    if units is None:
        _G = 1.
    else:
        _G = G.decompose(units).value

    def func(x,m,c):
        r = np.sqrt(np.sum(x**2, axis=-1))
        val = -_G * m / (r + c)
        return val

    def gradient(x,m,c):
        r = np.sqrt(np.sum(x**2, axis=-1))[...,None]

        fac = _G*m / ((r + c)**2 * r)
        return fac*x

    return func, gradient, None

class HernquistPotential(Potential):
    r"""
    Represents the Hernquist potential (1990) for a spheroid (bulge).

    .. math::

        \Phi_{spher} = -\frac{GM_{spher}}{r + c}

    Parameters
    ----------
    m : numeric
        Mass.
    c : numeric
        Core concentration.
    usys : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, c, usys=None):
        parameters = dict(m=m, c=c)
        func,gradient,hessian = hernquist_funcs(usys)
        super(HernquistPotential, self).__init__(func=func,
                                                 gradient=gradient,
                                                 hessian=hessian,
                                                 parameters=parameters)

##############################################################################
#    Triaxial, Logarithmic potential (see: Johnston et al. 1998)
#    http://adsabs.harvard.edu/abs/1999ApJ...512L.109J
#
def triaxial_log_funcs(units):

    def func(xyz, v_c, r_h, q1, q2, q3, phi):
        C1 = (math.cos(phi)/q1)**2+(math.sin(phi)/q2)**2
        C2 = (math.cos(phi)/q2)**2+(math.sin(phi)/q1)**2
        C3 = 2.*math.sin(phi)*math.cos(phi)*(1./q1**2 - 1./q2**2)

        x,y,z = xyz.T
        return 0.5*v_c*v_c * np.log(C1*x*x + C2*y*y + C3*x*y + z*z/q3**2 + r_h**2)

    def gradient(xyz, v_c, r_h, q1, q2, q3, phi):
        C1 = (math.cos(phi)/q1)**2+(math.sin(phi)/q2)**2
        C2 = (math.cos(phi)/q2)**2+(math.sin(phi)/q1)**2
        C3 = 2.*math.sin(phi)*math.cos(phi)*(1./q1**2 - 1./q2**2)

        x,y,z = xyz.T

        fac = 0.5*v_c*v_c / (C1*x*x + C2*y*y + C3*x*y + z*z/q3**2 + r_h**2)

        dx = fac * (2.*C1*x + C3*y)
        dy = fac * (2.*C2*y + C3*x)
        dz = 2.* fac * z / q3 / q3

        return np.array([dx,dy,dz]).T

    return func, gradient, None

class TriaxialLogarithmicPotential(Potential):
    r"""
    Represents a triaxial Logarithmic potential (e.g. triaxial halo).

    .. math::

        \Phi_{halo} = \frac{1}{2}v_{c}^2\ln(C1x^2 + C2y^2 + C3xy + z^2/q_3^2 + r_h^2)

    Parameters
    ----------
    v_c : numeric
        Circular velocity.
    r_h : numeric
        Scale radius.
    q1 : numeric
        Flattening in X-Y plane.
    q2 : numeric
        Flattening in X-Y plane.
    q3 : numeric
        Flattening in Z direction.
    phi : numeric
        Rotation of halo in X-Y plane.
    usys : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, v_c, r_h, q1, q2, q3, phi, usys):
        parameters = dict(v_c=v_c, r_h=r_h, q1=q1,
                          q2=q2, q3=q3, phi=phi)
        func,gradient,hessian = triaxial_log_funcs(usys)
        super(TriaxialLogarithmicPotential, self).__init__(func=func,
                                                 gradient=gradient,
                                                 hessian=hessian,
                                                 parameters=parameters)

# TODO: BELOW HERE

##############################################################################
#    Axisymmetric NFW potential
#
def _cartesian_axisymmetric_nfw_model(bases):
    """ Generates functions to evaluate an NFW potential and its
        derivatives at a specified position.

        Physical parameters for this potential are:
            m : total mass in the potential
            qz : z axis flattening
            Rs : scale-length
    """

    # scale G to be in this unit system
    _G = G.decompose(bases=bases).value

    def f(r,r_0,log_m,qz,Rs):
        rr = r-r_0
        x,y,z = rr.T

        m = np.exp(log_m)
        R_sq = x**2 + y**2
        sqrt_term = np.sqrt(R_sq + (z/qz)**2)
        val = -_G * m / sqrt_term * np.log(1. + sqrt_term/Rs)

        return val

    def df(r,r_0,log_m,qz,Rs):
        rr = r-r_0
        x,y,z = rr.T

        m = np.exp(log_m)
        zz = z/qz
        R = np.sqrt(x*x + y*y + zz*zz)

        term1 = 1./(R*R*(Rs+R))
        term2 = -np.log(1. + R/Rs) / (R*R*R)
        fac = _G*m * (term1 + term2)
        _x = fac * x
        _y = fac * y
        _z = fac * z / qz**2

        return np.array([_x,_y,_z]).T

    return (f, df)

class AxisymmetricNFWPotential(Potential):

    def __init__(self, units, **parameters):

        latex = "$\sigma$"

        assert "log_m" in parameters.keys(), "You must specify a log-mass."
        assert "qz" in parameters.keys(), "You must specify the parameter 'qz'."
        assert "Rs" in parameters.keys(), "You must specify the parameter 'Rs'."

        # get functions for evaluating potential and derivatives
        f,df = _cartesian_axisymmetric_nfw_model(units)
        super(AxisymmetricNFWPotential, self).__init__(units,
                                                 f=f, f_prime=df,
                                                 latex=latex,
                                                 parameters=parameters)

##############################################################################
#    Spherical NFW potential
#
def _cartesian_spherical_nfw_model(bases):
    """ Generates functions to evaluate an NFW potential and its
        derivatives at a specified position.

        Physical parameters for this potential are:
            log_m : total mass in the core of the potential
            a : scale-length
    """

    # scale G to be in this unit system
    _G = G.decompose(bases=bases).value

    def f(r,r_0,log_m,a):
        rr = r-r_0
        x,y,z = rr.T

        m = np.exp(log_m)
        R2 = x**2 + y**2 + z**2

        return -3 * _G * m / R2 * np.log(1. + R2/a)

    def df(r,r_0,log_m,a):
        rr = r-r_0
        x,y,z = rr.T

        m = np.exp(log_m)
        R = np.sqrt(x*x + y*y + z*z)

        dphi_dr = 3*_G*m/R * (np.log(1+R/a)/R - 1/(R+a))
        _x = dphi_dr * x/R
        _y = dphi_dr * y/R
        _z = dphi_dr * z/R

        return -np.array([_x,_y,_z]).T

    return (f, df)

class SphericalNFWPotential(Potential):

    def __init__(self, units, **parameters):

        latex = "$\sigma$"

        assert "log_m" in parameters.keys(), "You must specify a log-mass."
        assert "a" in parameters.keys(), "You must specify the parameter 'a'."

        # get functions for evaluating potential and derivatives
        f,df = _cartesian_spherical_nfw_model(units)
        super(SphericalNFWPotential, self).__init__(units,
                                                 f=f, f_prime=df,
                                                 latex=latex,
                                                 parameters=parameters)

    def v_circ(self, r):
        """ Return the circular velocity at position r """
        a = np.sqrt(np.sum(self.acceleration_at(r)**2, axis=-1))
        R = np.sqrt(np.sum(r**2, axis=-1))
        return np.sqrt(R*a)

##############################################################################
#    Axisymmetric, Logarithmic potential
#
def _cartesian_axisymmetric_logarithmic_model(bases):
    """ Generates functions to evaluate a Logarithmic potential and its
        derivatives at a specified position.

        Physical parameters for this potential are:
            qz : z-axis flattening parameter
            v_c : circular velocity of the halo
    """

    def f(r,r_0,v_c,qz):
        rr = r-r_0
        x,y,z = rr.T

        return 0.5*v_c*v_c * np.log(x*x + y*y + z*z/qz**2)

    def df(r,r_0,v_c,qz):
        rr = r-r_0
        x,y,z = rr.T

        fac = v_c*v_c / (x*x + y*y + z*z/(qz*qz))

        dx = fac * x
        dy = fac * y
        dz = fac * z / (qz*qz)

        return -np.array([dx,dy,dz]).T

    return (f, df)

class AxisymmetricLogarithmicPotential(Potential):

    def __init__(self, units, **parameters):
        """ Represents an axisymmetric Logarithmic potential

            $\Phi_{halo} = v_{c}^2/2\ln(x^2 + y^2 + z^2/q_z^2)$

            Model parameters: v_c, qz

            Parameters
            ----------
            units : list
                Defines a system of physical base units for the potential.
            parameters : dict
                A dictionary of parameters for the potential definition.
        """

        latex = "$\\Phi_{halo} = v_{halo}^2\\ln(C_1x^2 + C_2y^2 + C_3xy + z^2/q_z^2 + R_halo^2)$"

        for p in ["qz", "v_c"]:
            assert p in parameters.keys(), \
                    "You must specify the parameter '{0}'.".format(p)

        # get functions for evaluating potential and derivatives
        f,df = _cartesian_axisymmetric_logarithmic_model(units)
        super(AxisymmetricLogarithmicPotential, self).__init__(units,
                                                     f=f, f_prime=df,
                                                     latex=latex,
                                                     parameters=parameters)