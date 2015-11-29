# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy.constants import G

from .core import PotentialBase
from ..util import atleast_2d

__all__ = ["HarmonicOscillatorPotential", "KuzminPotential"]

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

    def _value(self, x, t, omega):
        return np.sum(0.5*atleast_2d(omega**2, insert_axis=1)*x**2, axis=0)

    def _gradient(self, x, t, omega):
        return atleast_2d(omega**2, insert_axis=1)*x

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

    def _value(self, q, t, m, a):
        x,y,z = q
        val = -self.G * m / np.sqrt(x**2 + y**2 + (a + np.abs(z))**2)
        return val

    def _gradient(self, q, t, m, a):
        x,y,z = q
        fac = self.G * m / (x**2 + y**2 + (a + np.abs(z))**2)**1.5
        return fac[None,...] * q
