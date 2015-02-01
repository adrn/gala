# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy.constants import G

from .core import Potential

__all__ = ["PointMassPotential", "IsochronePotential", "HarmonicOscillatorPotential",
           "KuzminPotential"]

# ============================================================================
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

class HarmonicOscillatorPotential(Potential):
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

# ============================================================================
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

# ============================================================================
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

class IsochronePotential(Potential):
    r"""
    The Isochrone potential.

    .. math::

        \Phi = -\frac{GM}{\sqrt{r^2+b^2} + b}

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

# ============================================================================
#    Kuzmin potential
#

def kuzmin_funcs(units):
    # scale G to be in this unit system
    if units is None:
        _G = 1.
    else:
        _G = G.decompose(units).value

    def func(q, m, a):
        x,y,z = q.T
        val = -_G * m / np.sqrt(x**2 + y**2 + (a + np.abs(z))**2)
        return val

    def gradient(q, m, a):
        x,y,z = q.T
        fac = _G * m / (x**2 + y**2 + (a + np.abs(z))**2)**1.5
        return fac[...,None] * q

    def hessian(x, m, a):
        raise NotImplementedError()  # TODO:

    return func, gradient, None

class KuzminPotential(Potential):
    r"""
    The Kuzmin flattened disk potential.

    .. math::

        \Phi = -\frac{Gm}{\sqrt{x^2 + y^2 + (a + |z|)^2}}

    Parameters
    ----------
    m : numeric
        Mass.
    a : numeric
        Flattening parameter.
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, a, units):
        parameters = dict(m=m, a=a)
        func,gradient,hessian = kuzmin_funcs(units)
        super(KuzminPotential, self).__init__(func=func, gradient=gradient,
                                              hessian=hessian,
                                              parameters=parameters, units=units)
