# coding: utf-8

from __future__ import division, print_function


# Standard library
from collections import OrderedDict

# Third-party
import numpy as np

from ..core import PotentialBase
from ....util import atleast_2d

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
        parameters = OrderedDict()
        parameters['omega'] = np.atleast_1d(omega)
        super(HarmonicOscillatorPotential, self).__init__(units=units,
                                                          parameters=parameters,
                                                          ndim=len(parameters['omega']))

    def _energy(self, q, t=0.):
        om = np.atleast_1d(self.parameters['omega'].value)
        return np.sum(0.5 * om[None]**2 * q**2, axis=1)

    def _gradient(self, q, t=0.):
        om = np.atleast_1d(self.parameters['omega'].value)
        return om[None]**2 * q

    def _hessian(self, q, t=0.):
        om = np.atleast_1d(self.parameters['omega'].value)
        return np.tile(np.diag(om)[:,:,None], reps=(1,1,q.shape[0]))

    def action_angle(self, w):
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
        w : :class:`gala.dynamics.PhaseSpacePosition`, :class:`gala.dynamics.Orbit`
            The positions or orbit to compute the actions, angles, and frequencies at.
        """
        from ....dynamics.analyticactionangle import harmonic_oscillator_to_aa
        return harmonic_oscillator_to_aa(w, self)

    # def phase_space(self, actions, angles):
    #     """
    #     Transform the input action-angle coordinates to cartesian position and velocity
    #     assuming a Harmonic Oscillator potential. This transformation
    #     is analytic and can be used as a "toy potential" in the
    #     Sanders & Binney 2014 formalism for computing action-angle coordinates
    #     in _any_ potential.

    #     Adapted from Jason Sanders' code
    #     `genfunc <https://github.com/jlsanders/genfunc>`_.

    #     Parameters
    #     ----------
    #     x : array_like
    #         Positions.
    #     v : array_like
    #         Velocities.
    #     """
    #     from ...dynamics.analyticactionangle import harmonic_oscillator_aa_to_xv
    #     return harmonic_oscillator_aa_to_xv(actions, angles, self)

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
        parameters = OrderedDict()
        parameters['m'] = m
        parameters['a'] = a
        super(KuzminPotential, self).__init__(units=units,
                                              parameters=parameters)

    def _energy(self, q, t):
        x,y,z = q
        m = self.parameters['m']
        a = self.parameters['a']
        val = -self.G * m / np.sqrt(x**2 + y**2 + (a + np.abs(z))**2)
        return val

    def _gradient(self, q, t):
        x,y,z = q
        m = self.parameters['m']
        a = self.parameters['a']
        fac = self.G * m / (x**2 + y**2 + (a + np.abs(z))**2)**1.5
        return fac[None,...] * q
