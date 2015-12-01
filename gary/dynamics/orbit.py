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
from ..util import inherit_docs, atleast_2d

__all__ = ['CartesianPhaseSpacePosition', 'CartesianOrbit']

class PhaseSpacePosition(object):
    pass

class CartesianPhaseSpacePosition(PhaseSpacePosition):

    def __init__(self, pos, vel):

        # make sure position and velocity input are 2D
        pos = atleast_2d(pos, insert_axis=1)
        vel = atleast_2d(vel, insert_axis=1)

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

    def __repr__(self):
        return "<CartesianPhaseSpacePosition {}>".format(self.pos.shape)

    def __str__(self):
        return "CartesianPhaseSpacePosition"

    def __getitem__(self, slyce):
        try:
            _slyce = (slice(None),) + tuple(slyce)
        except TypeError:
            _slyce = (slice(None),) + (slyce,)

        return self.__class__(pos=self.pos[_slyce], vel=self.vel[_slyce])

    @property
    def ndim(self):
        """
        Number of coordinate dimensions. 1/2 of the phase-space dimensionality.

        .. warning::

            This is *not* the number of axes in the position or velocity
            arrays. That is accessed by doing ``orbit.pos.ndim``.
        """
        return self.pos.shape[0]

    # ------------------------------------------------------------------------
    # Convert from Cartesian to other representations
    # ------------------------------------------------------------------------
    def represent_as(self, Representation):
        """
        Represent the position and velocity of the orbit in an alternate
        coordinate system. Supports any of the Astropy coordinates
        representation classes.

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
        car_pos = coord.CartesianRepresentation(self.pos)
        new_pos = car_pos.represent_as(Representation)

        # transform the velocity
        vfunc = getattr(vtrans, "cartesian_to_{}".format(rep_name))
        new_vel = vfunc(car_pos, self.vel)

        return new_pos, new_vel

    def to_frame(self, frame, galactocentric_frame=coord.Galactocentric(),
                 vcirc=None, vlsr=None):
        """
        Transform the orbit from Galactocentric, cartesian coordinates to
        Heliocentric coordinates in the specified Astropy coordinate frame.

        Parameters
        ----------
        frame : :class:`~astropy.coordinates.BaseCoordinateFrame`
        galactocentric_frame : :class:`~astropy.coordinates.Galactocentric`
        vcirc : :class:`~astropy.units.Quantity`
            TODO. Passed to velocity transformation.
        vlsr : :class:`~astropy.units.Quantity`
            TODO. Passed to velocity transformation.

        Returns
        -------
        c : :class:`~astropy.coordinates.BaseCoordinateFrame`
        v : tuple

        """

        car_pos = coord.CartesianRepresentation(self.pos)
        gc_c = galactocentric_frame.realize_frame(car_pos)
        c = gc_c.transform_to(frame)

        kw = dict()
        kw['galactocentric_frame'] = galactocentric_frame
        if vcirc is not None:
            kw['vcirc'] = vcirc
        if vlsr is not None:
            kw['vlsr'] = vlsr

        v = vgal_to_hel(c, self.vel, **kw)
        return c, v

    # ------------------------------------------------------------------------
    # Computed dynamical quantities
    # ------------------------------------------------------------------------
    def kinetic_energy(self, potential):
        """
        The kinetic energy. This is currently *not* cached and is
        computed each time the attribute is accessed.
        """
        pass
        # TODO: waiting until I overhaul how potentials handle units...

    def potential_energy(self, potential):
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

    def energy(self, potential):
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

    # ------------------------------------------------------------------------
    # Misc. useful methods
    # ------------------------------------------------------------------------
    def plot(self):
        pass

# ----------------------------------------------------------------------------

class Orbit(object):
    pass

@inherit_docs
class CartesianOrbit(CartesianPhaseSpacePosition, Orbit):
    """
    Represents an orbit in Cartesian coordinates -- positions
    and velocities (conjugate momenta) at different times.

    .. warning::

        This is an experimental class. The API can and probably will change!

    The position and velocity quantities (arrays) can have an arbitrary
    number of dimensions, but the first two axes (0, 1) have
    special meaning::

        - `axis=0` is the coordinatte dimension (e.g., x, y, z)
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

        super(CartesianOrbit, self).__init__(pos=pos, vel=vel)

        if t is not None:
            t = np.atleast_1d(t)
            if self.pos.shape[1] != len(t):
                raise ValueError("Position and velocity must have the same length "
                                 "along axis=1 as the length of the time array "
                                 "{} vs {}".format(len(t), self.pos.shape[1]))

            if not hasattr(t, 'unit'):
                t = t * uno

        self.t = t
        self.potential = potential

    def __repr__(self):
        return "<Orbit {}>".format(self.pos.shape)

    def __str__(self):
        return "Orbit"

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

    def w(self, units=None):
        """
        This returns a single array containing the phase-space positions.
        If no unit system is specified, this tries to use the unit system
        defined by the potential associated with this orbit. An error is
        thrown if no units are given and no potential is set.

        Parameters
        ----------
        units : `gary.units.UnitSystem` (optional)
            TODO

        Returns
        -------
        w : `~numpy.ndarray`
            TODO

        """
        if self.pos.unit == uno and self.vel.unit == uno:
            units = [uno]

        else:
            if units is None and self.potential is None:
                raise ValueError("If no UnitSystem is specified, the orbit must have "
                                 "an associated potential object.")

            if units is None and self.potential.units is None:
                raise ValueError("If no UnitSystem is specified, the potential object "
                                 "associated with this orbit must have a UnitSystem defined.")

            if units is None:
                units = self.potential.units

        x = self.pos.decompose(units).value
        v = self.vel.decompose(units).value
        return np.vstack((x,v))

    # ------------------------------------------------------------------------
    # Computed dynamical quantities
    # ------------------------------------------------------------------------
    @property
    def kinetic_energy(self):
        """
        The kinetic energy. This is currently *not* cached and is
        computed each time the attribute is accessed.
        """
        return super(CartesianOrbit,self).kinetic_energy(self.potential)

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
        return super(CartesianOrbit,self).potential_energy(self.potential)

    @property
    def energy(self):
        """
        The total energy (kinetic + potential). This is currently *not*
        cached and is computed each time the attribute is accessed.
        """
        return super(CartesianOrbit,self).energy(self.potential)

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
            r = np.sqrt(np.sum(self.pos**2, axis=0))
            T = [peak_to_peak_period(self.t.value, r[i]) for i in range(r.shape[0])]
            T = T * self.t.unit

        else:
            # TODO: this is broken
            T = np.zeros(self.pos.shape[[]])
            for i in range(self.pos.shape[1]):
                for k in range(self.pos.shape[-1]):
                    T[i,k] = peak_to_peak_period(self.t.value, self.pos[:,i,k])
            T = T * self.t.unit

        return T

    # ------------------------------------------------------------------------
    # Misc. useful methods
    # ------------------------------------------------------------------------
    def plot(self):
        pass

    def align_circulation_with_z(self):
        # TODO: returns copy
        pass
