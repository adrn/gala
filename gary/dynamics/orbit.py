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
from .core import angular_momentum, peak_to_peak_period
from ..coordinates import velocity_transforms as vtrans
from ..coordinates import vgal_to_hel

__all__ = ['CartesianOrbit']

class CartesianOrbit(object):
    """
    Represents an orbit in Cartesian coordinates -- positions
    and velocities (conjugate momenta) at different times.

    .. warning::

        This is an experimental class. The API can and probably will change!

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
        TODO

        Parameters
        ----------
        Representation : :class:`~astropy.coordinates.BaseRepresentation`
            The class for the desired representation.

        Returns
        -------
        pos : `~astropy.coordinates.BaseRepresentation`
        vel : `~astropy.units.Quantity`
            The velocity in the new representation. All components
            have units of velocity -- e.g., to get an angular velocity
            in spherical representations, you'll need to divide by the radius.
        """
        # get the name of the desired representation
        rep_name = Representation.get_name()

        # transform the position
        car_pos = coord.CartesianRepresentation(self.pos.T)
        new_pos = car_pos.represent_as(Representation)

        # transform the velocity
        vfunc = getattr(vtrans, "cartesian_to_{}".format(rep_name))
        new_vel = vfunc(car_pos, self.vel.T)

        return new_pos, new_vel.T

    def to_frame(self, frame, galactocentric_frame=coord.Galactocentric(), **kwargs):
        """
        TODO

        Parameters
        ----------
        frame : :class:`~astropy.coordinates.BaseCoordinateFrame`
        galactocentric_frame : :class:`~astropy.coordinates.Galactocentric`
        **kwargs
            Passed to velocity transform.

        Returns
        -------
        c : :class:`~astropy.coordinates.BaseCoordinateFrame`
        v : tuple

        """

        car_pos = coord.CartesianRepresentation(self.pos.T)
        gc_c = galactocentric_frame.realize_frame(car_pos)
        c = gc_c.transform_to(frame)

        v = vgal_to_hel(c, self.vel.T, galactocentric_frame=galactocentric_frame, **kwargs)
        return c, [v[i].T for i in range(3)]

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

    def estimate_period(self, radial=True):
        """
        Estimate the period of the orbit. By default, computes the radial
        period. If ``radial==False``, this returns period estimates for
        each dimension of the orbit.

        Parameters
        ----------
        radial : bool (optional)
            What period to estimate. If ``True``, estimates the radial
            period. If ``False``, estimates period in each dimension, e.g.,
            if the orbit is 3D, along x, y, and z.

        Returns
        -------
        T : `~astropy.units.Quantity`
            The period or periods.
        """

        if self.t is None:
            raise ValueError("To compute the period, a time array is needed."
                             " Specify a time array when creating this object.")

        if radial:
            r = np.sqrt(np.sum(self.pos**2, axis=-1))
            T = [peak_to_peak_period(self.t.value, r[i]) for i in range(r.shape[1])]
            T = T * self.t.unit

        else:
            T = np.zeros(self.pos.shape[1:])
            for i in range(self.pos.shape[1]):
                for k in range(self.pos.shape[-1]):
                    T[i,k] = peak_to_peak_period(self.t.value, self.pos[:,i,k])
            T = T * self.t.unit

        return T

    # Methods
    def plot(self):
        pass

    def align_circulation_with_z(self):
        # TODO: returns copy
        pass
