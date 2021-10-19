# Third-party
import numpy as np

from gala.potential.potential.core import PotentialBase
from gala.potential.potential.util import sympy_wrap
from gala.potential.common import PotentialParameter

__all__ = ["HarmonicOscillatorPotential"]


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
    omega = PotentialParameter('omega', physical_type='frequency')

    def _setup_potential(self, parameters, origin=None, R=None, units=None):
        parameters['omega'] = np.atleast_1d(parameters['omega'])
        super()._setup_potential(parameters, origin=origin, R=R, units=units)
        self.ndim = len(self.parameters['omega'])

    def _energy(self, q, t=0.):
        om = np.atleast_1d(self.parameters['omega'].value)
        return np.sum(0.5 * om[None]**2 * q**2, axis=1)

    def _gradient(self, q, t=0.):
        om = np.atleast_1d(self.parameters['omega'].value)
        return om[None]**2 * q

    def _hessian(self, q, t=0.):
        om = np.atleast_1d(self.parameters['omega'].value)
        return np.tile(np.diag(om)[:, :, None], reps=(1, 1, q.shape[0]))

    @classmethod
    @sympy_wrap(var='x')
    def to_sympy(cls, v, p):
        expr = 1/2 * p['omega']**2 * v['x']**2
        return expr, v, p

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
        from gala.dynamics.actionangle import harmonic_oscillator_xv_to_aa
        return harmonic_oscillator_xv_to_aa(w, self)

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
    #     from gala.dynamics.actionangle import harmonic_oscillator_aa_to_xv
    #     return harmonic_oscillator_aa_to_xv(actions, angles, self)
