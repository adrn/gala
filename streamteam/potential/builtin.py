# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import math

# Third-party
import numpy as np
from astropy.constants import G

from .core import Potential, CartesianPotential

__all__ = ["PointMassPotential", "MiyamotoNagaiPotential",
           "HernquistPotential", "LogarithmicPotential",
           "IsochronePotential", "HarmonicOscillatorPotential"]

############################################################
#    Harmonic oscillator
#
def harmonic_osc_funcs(units):
    def f(x, omega):
        omega2 = omega*omega
        return np.sum(0.5*np.atleast_2d(omega2)*x*x, axis=-1)

    def gradient(x, omega):
        omega2 = omega*omega
        return np.atleast_2d(omega2)*x

    def hessian(x, x0, m):
        raise NotImplementedError()  # TODO:

    return f, gradient, None

class HarmonicOscillatorPotential(CartesianPotential):
    r"""
    Represents an N-dimensional harmonic oscillator.

    .. math::

        \Phi = \frac{1}{2}\omega^2 x^2

    Parameters
    ----------
    omega : numeric
        Frequency.
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.
    """

    def __init__(self, omega, units=None):
        parameters = dict(omega=np.array(omega))
        func,gradient,hessian = harmonic_osc_funcs(units)
        super(HarmonicOscillatorPotential, self).__init__(func=func, gradient=gradient,
                                                          hessian=hessian,
                                                          parameters=parameters, units=units)

    def action_angle(self, x, v):
        """
        Transform the input cartesian position and velocity to action-angle
        coordinates the Harmonic Oscillator potential. This transformation
        is analytic and can be used as a "toy potential" in the
        Sanders & Binney 2014 formalism for computing action-angle coordinates
        in _any_ potential.

        Adapted from Jason Sanders' code
        `genfunc <https://github.com/jlsanders/genfunc>`_.

        Parameters
        ----------
        x : array_like
            Positions.
        v : array_like
            Velocities.
        """
        from ..dynamics.analyticactionangle import harmonic_oscillator_xv_to_aa
        return harmonic_oscillator_xv_to_aa(x, v, self)

    def phase_space(self, actions, angles):
        """
        Transform the input action-angle coordinates to cartesian position and velocity
        assuming a Harmonic Oscillator potential. This transformation
        is analytic and can be used as a "toy potential" in the
        Sanders & Binney 2014 formalism for computing action-angle coordinates
        in _any_ potential.

        Adapted from Jason Sanders' code
        `genfunc <https://github.com/jlsanders/genfunc>`_.

        Parameters
        ----------
        x : array_like
            Positions.
        v : array_like
            Velocities.
        """
        from ..dynamics.analyticactionangle import harmonic_oscillator_aa_to_xv
        return harmonic_oscillator_aa_to_xv(actions, angles, self)

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
        a = (np.sum(xx**2, axis=-1)**-1.5)[...,None]
        return _G * m * xx * a

    def hessian(x, x0, m):
        raise NotImplementedError()  # TODO:

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
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.
    """

    def __init__(self, m, x0, units=None):
        parameters = dict(m=m, x0=x0)
        func,gradient,hessian = point_mass_funcs(units)
        super(PointMassPotential, self).__init__(func=func, gradient=gradient,
                                                 hessian=hessian,
                                                 parameters=parameters, units=units)

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
        raise NotImplementedError()  # TODO:

    return func, gradient, None

class IsochronePotential(CartesianPotential):
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
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, b, units):
        parameters = dict(m=m, b=b)
        func,gradient,hessian = isochrone_funcs(units)
        super(IsochronePotential, self).__init__(func=func, gradient=gradient,
                                                 hessian=hessian,
                                                 parameters=parameters, units=units)

    def action_angle(self, x, v):
        """
        Transform the input cartesian position and velocity to action-angle
        coordinates the Isochrone potential. See Section 3.5.2 in
        Binney & Tremaine (2008), and be aware of the errata entry for
        Eq. 3.225.

        This transformation is analytic and can be used as a "toy potential"
        in the Sanders & Binney 2014 formalism for computing action-angle
        coordinates in _any_ potential.

        Adapted from Jason Sanders' code
        `here <https://github.com/jlsanders/genfunc>`_.

        Parameters
        ----------
        x : array_like
            Positions.
        v : array_like
            Velocities.
        """
        from ..dynamics.analyticactionangle import isochrone_xv_to_aa
        return isochrone_xv_to_aa(x, v, self)

    def phase_space(self, actions, angles):
        """
        Transform the input actions and angles to ordinary phase space (position
        and velocity) in cartesian coordinates. See Section 3.5.2 in
        Binney & Tremaine (2008), and be aware of the errata entry for
        Eq. 3.225.

        Parameters
        ----------
        actions : array_like
        angles : array_like
        """
        from ..dynamics.analyticactionangle import isochrone_aa_to_xv
        return isochrone_aa_to_xv(actions, angles, self)

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

    def hessian(xyz, m, a, b):  # TODO
        pass

    return func, gradient, None

class MiyamotoNagaiPotential(CartesianPotential):
    r"""
    Miyamoto-Nagai potential for a flattened mass distribution.

    .. math::

        \Phi_{disk} = -\frac{GM_{disk}}{\sqrt{R^2 + (a + \sqrt{z^2 + b^2})^2}}

    Parameters
    ----------
    m : numeric
        Mass.
    a : numeric
    b : numeric
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, a, b, units):
        parameters = dict(m=m, a=a, b=b)
        func,gradient,hessian = miyamoto_nagai_funcs(units)
        super(MiyamotoNagaiPotential, self).__init__(func=func,
                                                     gradient=gradient,
                                                     hessian=hessian,
                                                     parameters=parameters, units=units)

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

class HernquistPotential(CartesianPotential):
    r"""
    Hernquist potential for a spheroid.

    .. math::

        \Phi_{spher} = -\frac{GM_{spher}}{r + c}

    Parameters
    ----------
    m : numeric
        Mass.
    c : numeric
        Core concentration.
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, c, units):
        parameters = dict(m=m, c=c)
        func,gradient,hessian = hernquist_funcs(units)
        super(HernquistPotential, self).__init__(func=func,
                                                 gradient=gradient,
                                                 hessian=hessian,
                                                 parameters=parameters, units=units)

##############################################################################
#    Triaxial, Logarithmic potential (see: Johnston et al. 1998)
#    http://adsabs.harvard.edu/abs/1999ApJ...512L.109J
#
def log_funcs(units):

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

class LogarithmicPotential(CartesianPotential):
    r"""
    Triaxial logarithmic potential.

    .. math::

        \Phi &= \frac{1}{2}v_{c}^2\ln(C_1x^2 + C_2y^2 + C_3xy + z^2/q_3^2 + r_h^2)\\
        C_1 &= \frac{\cos^2\phi}{q_1^2} + \frac{\sin^2\phi}{q_2^2}\\
        C_2 &= \frac{\sin^2\phi}{q_1^2} + \frac{\cos^2\phi}{q_2^2}\\
        C_3 &= 2\sin\phi\cos\phi \left(q_1^{-2} - q_2^{-2}\right)

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
    units : iterable (optional)
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, v_c, r_h, q1, q2, q3, phi, units=None):
        parameters = dict(v_c=v_c, r_h=r_h, q1=q1,
                          q2=q2, q3=q3, phi=phi)
        func,gradient,hessian = log_funcs(units)
        super(LogarithmicPotential, self).__init__(func=func,
                                                   gradient=gradient,
                                                   hessian=hessian,
                                                   parameters=parameters, units=units)
