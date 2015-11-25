# coding: utf-8

""" General dynamics utilities. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy import log as logger
import astropy.units as u
uno = u.dimensionless_unscaled
import astropy.coordinates as coord
import numpy as np

# Project
from .core import angular_momentum

__all__ = ['CartesianOrbit']

class CartesianOrbit(object):
    """
    Represents an orbit in Cartesian coordinates -- positions
    and velocities (conjugate momenta) at different times.

    The position and velocity quantities (arrays) can have an arbitrary
    number of dimensions, but the first and last axes (0 and -1) have
    special meaning::

        - `axis=0` is the time dimension
        - `axis=-1` is the coordinate dimension (e.g., x, y, z)

    So if the input position array, `pos`, has shape `pos.shape = (100, 3)`, this
    would be a 3D orbit (`pos[...,0]` is `x`, `pos[...,1]` is `y`, etc.). For representing
    multiple orbits, the position array could have 3 axes, e.g., it might have shape
    `pos.shape = (100, 8, 3)`, where this is interpreted as a 3D position at 100 times
    for 8 different orbits. The same is true for velocity. The position and velocity
    arrays must have the same shape.

    If a time argument is specified, the position and velocity arrays must have
    the same number of timesteps as the length of the time object::

        len(t) == pos.shape[0]

    Parameters
    ----------
    pos : array_like, :class:`~astropy.units.Quantity`
        Positions. If a numpy array (e.g., has no units), this will be
        stored as a dimensionless :class:`~astropy.units.Quantity`. See
        the note above about the assumed meaning of the axes of this object.
    vel : array_like, :class:`~astropy.units.Quantity`
        Velocities. If a numpy array (e.g., has no units), this will be
        stored as a dimensionless :class:`~astropy.units.Quantity`. See
        the note above about the assumed meaning of the axes of this object.
    t : array_like, :class:`~astropy.units.Quantity` (optional)
        Array of times. If a numpy array (e.g., has no units), this will be
        stored as a dimensionless :class:`~astropy.units.Quantity`.
    potential : `~gary.potential.PotentialBase` (optional)
        The potential that the orbit was integrated in.

    """
    def __init__(self, pos, vel, t=None, potential=None):

        # make sure position and velocity input are 2D
        pos = np.atleast_2d(pos)
        vel = np.atleast_2d(vel)

        if t is not None:
            t = np.atleast_1d(t)
            if pos.shape[0] != len(t):
                raise ValueError("Position and velocity must have the same length "
                                 "along axis=0 as the length of the time array "
                                 "{} vs {}".format(len(t), pos.shape[0]))

            if not hasattr(t, 'unit'):
                t = t * uno

        # make sure position and velocity have at least a dimensionless unit!
        if not hasattr(pos, 'unit'):
            pos = pos * uno

        if not hasattr(vel, 'unit'):
            vel = vel * uno

        if (pos.unit == uno and vel.unit != uno):
            logger.warning("Position unit is dimensionless but velocity unit is not."
                           "Are you sure that's what you want?")
        elif (vel.unit == uno and pos.unit != uno):
            logger.warning("Velocity unit is dimensionless but position unit is not."
                           "Are you sure that's what you want?")

        # make sure shape is the same
        for i in range(pos.ndim):
            if pos.shape[i] != vel.shape[i]:
                raise ValueError("Position and velocity must have the same shape "
                                 "{} vs {}".format(pos.shape, vel.shape))

        self.pos = pos
        self.vel = vel
        self.t = t
        self.potential = potential

    def __repr__(self):
        return "<Orbit {}>".format(self.pos.shape)

    def __str__(self):
        pass

    def __getitem__(self, slyce):
        try:
            _slyce = tuple(slyce) + (slice(None),)
        except TypeError:
            _slyce = (slyce,) + (slice(None),)

        kw = dict()
        if self.t is not None:
            kw['t'] = self.t[_slyce[0]]
        return self.__class__(pos=self.pos[_slyce], vel=self.vel[_slyce],
                              potential=self.potential, **kw)

    @property
    def ndim(self):
        """
        Number of coordinate dimensions.

        .. warning::

            This is *not* the number of axes in the position or velocity
            arrays. That is accessed by doing ``orbit.pos.ndim``.
        """
        return self.pos.shape[-1]

    # Convert from Cartesian to other representations
    def represent_as(self, Representation):
        """

        Parameters
        ----------
        Representation : :class:`~astropy.coordinates.BaseRepresentation`
            The class for the desired representation.
        """
        pass

    # Computed quantities
    @property
    def kinetic_energy(self):
        """
        The kinetic energy. This is currently *not* cached and is
        computed each time the attribute is accessed.
        """
        pass
        # TODO: waiting until I overhaul how potentials handle units...

    @property
    def potential_energy(self):
        """
        The potential energy. This is currently *not* cached and is
        computed each time the attribute is accessed.
        """
        if self.potential is None:
            raise ValueError("To compute the potential energy, a potential"
                             " object must be provided when creating the"
                             " orbit object!")
        pass
        # TODO: waiting until I overhaul how potentials handle units...

    @property
    def energy(self):
        """
        The total energy (kinetic + potential). This is currently *not*
        cached and is computed each time the attribute is accessed.
        """
        return self.kinetic_energy + self.potential_energy
        # TODO: waiting until I overhaul how potentials handle units...

    @property
    def angular_momentum(self):
        """
        The angular momentum. This is currently *not* cached and is
        computed each time the attribute is accessed.
        """
        return angular_momentum(self.pos, self.vel)

    # Methods
    def plot(self):
        pass

    def align_circulation_with_z(self):
        # TODO: returns copy
        pass
