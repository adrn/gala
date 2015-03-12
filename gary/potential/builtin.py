# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy.constants import G

from .core import PotentialBase

__all__ = ["PointMassPotential", "IsochronePotential", "HarmonicOscillatorPotential",
           "KuzminPotential"]

class HarmonicOscillatorPotential(PotentialBase):
    r"""
    Represents an N-dimensional harmonic oscillator.

    .. math::

        \Phi = \frac{1}{2}\omega^2 x^2

    Parameters
    ----------
    omega : numeric
        Frequency.
    units : iterable(optional)
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.
    """

    def __init__(self, omega, units=None):
        self.parameters = dict(omega=np.array(omega))
        super(HarmonicOscillatorPotential, self).__init__(units=units)

    def _value(self, x, omega):
        return np.sum(0.5*np.atleast_2d(omega**2)*x**2, axis=-1)

    def _gradient(self, x, omega):
        return np.atleast_2d(omega**2)*x

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


class PointMassPotential(PotentialBase):
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
        self.parameters = dict(m=m, x0=x0)
        self.units = units
        if units is not None:
            self.G = G.decompose(units).value
        else:
            self.G = 1.

    def _value(self, x, x0, m):
        xx = x-x0
        R = np.sqrt(np.sum(xx**2, axis=-1))
        return -self.G * m / R

    def _gradient(self, x, x0, m):
        xx = x-x0
        a = (np.sum(xx**2, axis=-1)**-1.5)[...,None]
        return self.G * m * xx * a


class IsochronePotential(PotentialBase):
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
        self.parameters = dict(m=m, b=b)
        self.units = units
        self.G = G.decompose(units).value

    def _value(self, x, m, b):
        r2 = np.sum(x**2, axis=-1)
        val = -self.G * m / (np.sqrt(r2 + b*b) + b)
        return val

    def _gradient(self, x, m, b):
        r2 = np.sum(x**2, axis=-1)
        fac = self.G*m / (np.sqrt(r2 + b*b) + b)**2 / np.sqrt(r2 + b*b)
        return fac[...,None] * x

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

class KuzminPotential(PotentialBase):
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
        self.parameters = dict(m=m, a=a)
        self.units = units
        self.G = G.decompose(units).value

    def _value(self, q, m, a):
        x,y,z = q.T
        val = -self.G * m / np.sqrt(x**2 + y**2 + (a + np.abs(z))**2)
        return val

    def _gradient(self, q, m, a):
        x,y,z = q.T
        fac = self.G * m / (x**2 + y**2 + (a + np.abs(z))**2)**1.5
        return fac[...,None] * q
