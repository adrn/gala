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
from ..util import atleast_2d

__all__ = ['CartesianOrbit']

class CartesianOrbit(object):
    """
    Represents an orbit in Cartesian coordinates -- positions
    and velocities (conjugate momenta) at different times.

    The position and velocity quantities (arrays) can have an arbitrary
    number of dimensions, but axis 0 and 1 have special meaning:

        - `axis=0` is the coordinate dimension (e.g., x, y, z)
        - `axis=1` is the time dimension

    So if the input position array, `pos`, has shape `pos.shape = (3, 100)`, this
    would be a 3D orbit (`pos[0]` is `x`, `pos[1]` is `y`, etc.). For representing
    multiple orbits, the position array could have 3 axes, e.g., it might have shape
    `pos.shape = (3, 100, 8)`, where this is interpreted as a 3D position at 100 times
    for 8 different orbits. The same is true for velocity. The position and velocity
    arrays must have the same shape.

    If a time argument is specified, the position and velocity arrays must have
    the same number of timesteps as the length of the time object::

        len(t) == pos.shape[1]

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
        pos = atleast_2d(pos, insert_axis=1)
        vel = atleast_2d(vel, insert_axis=1)

        if t is not None:
            t = np.atleast_1d(t)
            if pos.shape[1] != len(t):
                raise ValueError("Position and velocity must have the same length "
                                 "along axis=1 as the length of the time array "
                                 "{} vs {}".format(len(t), pos.shape[1]))

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
            _slyce = (slice(None),) + tuple(slyce)
        except TypeError:
            _slyce = (slice(None),) + (slyce,)

        kw = dict()
        if self.t is not None:
            kw['t'] = self.t[_slyce[1]]
        return self.__class__(pos=self.pos[_slyce], vel=self.vel[_slyce],
                              potential=self.potential, **kw)

    def __copy__(self):
        # TODO:
        pass

    def __deepcopy__(self):
        # TODO:
        pass

    def copy(self, deep=False):
        # TODO:
        pass

    def represent_as(self, Representation):
        """
        Transform the representation or coordinate system of the orbit, for example,
        from Cartesian to Spherical.

        Parameters
        ----------
        Representation : :class:`~astropy.coordinates.BaseRepresentation`
            The output representation class.
        """

        if self.Representation == Representation:
            return self

        # first transform the position
        new_pos = self.Representation(*self.pos).represent_as(Representation)

        # now find the function to transform the velocity
        _func_name = "{}_to_{}".format(self.Representation.get_name(),
                                       Representation.get_name())
        v_func = getattr(vtrans, _func_name)
        new_vel = v_func(self.pos, self.vel)

        # return Orbit(pos=new_pos, vel=new_vel, t=self.t, unitsys=self.unitsys,
        #              Representation=Representation, potential=self.potential)

    @property
    def shape(self):
        """
        Returns the shape of the orbit, without the implicit dimensionality axis=0
        (assumed to be 3D in position and velocity).

        """

        return self.pos.shape[1:]

    @property
    def norbits(self):
        if len(self.pos.shape) == 2:
            return 1
        else:
            return self.pos.shape[-1]

    @property
    def orbit_type(self):
        # TODO: figure out if tube or box (.orbit_type)
        pass

    def kinetic_energy(self):
        pass

    def total_energy(self):
        pass

    def align_circulation_with_z(self):
        # TODO: returns copy
        pass

    def plot(self):
        pass
