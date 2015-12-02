# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy import log as logger
import astropy.units as u
uno = u.dimensionless_unscaled
import numpy as np

# Project
from .core import CartesianPhaseSpacePosition
from .util import peak_to_peak_period
from .plot import plot_orbits
from ..util import inherit_docs, atleast_2d

__all__ = ['CartesianOrbit', 'combine']

class Orbit(object):
    pass

@inherit_docs
class CartesianOrbit(CartesianPhaseSpacePosition, Orbit):
    """
    Represents an orbit in Cartesian coordinates -- positions
    and velocities (conjugate momenta) at different times.

    .. warning::

        This is an experimental class. The API may change in a future release!

    The position and velocity quantities (arrays) can have an arbitrary
    number of dimensions, but the first two axes (0, 1) have
    special meaning::

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
    pos : :class:`~astropy.units.Quantity`, array_like
        Positions. If a numpy array (e.g., has no units), this will be
        stored as a dimensionless :class:`~astropy.units.Quantity`. See
        the note above about the assumed meaning of the axes of this object.
    vel : :class:`~astropy.units.Quantity`, array_like
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
        return "<CartesianOrbit N={}, shape={}>".format(self.ndim, self.pos.shape[1:])

    def __str__(self):
        return "CartesianOrbit"

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

    @property
    def norbits(self):
        if self.pos.ndim < 3:
            return 1
        else:
            return self.pos.shape[2]

    def w(self, units=None):
        """
        This returns a single array containing the phase-space positions.

        Parameters
        ----------
        units : `~gary.units.UnitSystem` (optional)
            The unit system to represent the position and velocity in
            before combining into the full array.

        Returns
        -------
        w : `~numpy.ndarray`
            A numpy array of all positions and velocities, without units.
            Will have shape ``(2*ndim,...)``.

        """
        if self.pos.unit == uno and self.vel.unit == uno and units is None:
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
        w = np.vstack((x,v))
        if w.ndim < 3:
            w = w[...,np.newaxis] # one orbit
        return w

    # ------------------------------------------------------------------------
    # Computed dynamical quantities
    # ------------------------------------------------------------------------

    def potential_energy(self, potential=None):
        r"""
        The potential energy *per unit mass*:

        .. math::

            E_\Phi = \Phi(\boldsymbol{q})

        Returns
        -------
        E : :class:`~astropy.units.Quantity`
            The potential energy.
        """
        if self.potential is None and potential is None:
            raise ValueError("To compute the potential energy, a potential"
                             " object must be provided!")
        if potential is None:
            potential = self.potential

        return super(CartesianOrbit,self).potential_energy(potential)

    def energy(self, potential=None):
        r"""
        The total energy *per unit mass* (e.g., kinetic + potential):

        Returns
        -------
        E : :class:`~astropy.units.Quantity`
            The total energy.
        """
        return self.kinetic_energy() + self.potential_energy(potential)

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
            r = np.sqrt(np.sum(self.pos**2, axis=0)).value
            if self.norbits == 1:
                T = peak_to_peak_period(self.t.value, r)
                T = T * self.t.unit
            else:
                T = [peak_to_peak_period(self.t.value, r[:,n]) for n in range(r.shape[1])]
                T = T * self.t.unit

        else:
            raise NotImplementedError("sorry 'bout that...")

        return T

    # ------------------------------------------------------------------------
    # Misc. useful methods
    # ------------------------------------------------------------------------
    def circulation(self):
        """
        Determine which axes the Orbit circulates around by checking
        whether there is a change of sign of the angular momentum
        about an axis. Returns a 2D array with ``ndim`` integers per orbit
        point. If a box orbit, all integers will be 0. A 1 indicates
        circulation about the corresponding axis.

        TODO: clockwise / counterclockwise?

        For example, for a single 3D orbit:

        - Box and boxlet = [0,0,0]
        - z-axis (short-axis) tube = [0,0,1]
        - x-axis (long-axis) tube = [1,0,0]

        Returns
        -------
        circulation : :class:`numpy.ndarray`
            An array that specifies whether there is circulation about any of
            the axes of the input orbit. For a single orbit, will return a
            1D array, but for multiple orbits, the shape will be ``(3, norbits)``.

        """
        L = self.angular_momentum()

        # if only 2D, add another empty axis
        if L.ndim == 2:
            single_orbit = True
            L = L[...,None]
        else:
            single_orbit = False

        ndim,ntimes,norbits = L.shape

        # initial angular momentum
        L0 = L[:,0]

        # see if at any timestep the sign has changed
        circ = np.ones((ndim,norbits))
        for ii in range(ndim):
            cnd = (np.sign(L0[ii]) != np.sign(L[ii,1:])) | \
                  (np.abs(L[ii,1:]).value < 1E-13)
            ix = np.atleast_1d(np.any(cnd, axis=0))
            circ[ii,ix] = 0

        circ = circ.astype(int)
        if single_orbit:
            return circ.reshape((ndim,))
        else:
            return circ

    def align_circulation_with_z(self, circulation=None):
        """
        If the input orbit is a tube orbit, this function aligns the circulation
        axis with the z axis and returns a copy.

        Parameters
        ----------
        circulation : array_like (optional)
            Array of bits that specify the axis about which the orbit circulates. If
            not provided, will compute this using
            :meth:`~gary.dynamics.CartesianOrbit.circulation`. See that method for
            more information.

        Returns
        -------
        orb : :class:`~gary.dynamics.CartesianOrbit`
            A copy of the original orbit object with circulation aligned with the z axis.
        """

        if circulation is None:
            circulation = self.circulation()
        circulation = atleast_2d(circulation, insert_axis=1)

        if self.pos.ndim < 3:
            pos = self.pos[...,np.newaxis]
            vel = self.vel[...,np.newaxis]
        else:
            pos = self.pos
            vel = self.vel

        if circulation.shape[0] != self.ndim or circulation.shape[1] != pos.shape[2]:
            raise ValueError("Shape of 'circulation' array should match the shape"
                             " of the position/velocity (minus the time axis).")

        new_pos = pos.copy()
        new_vel = vel.copy()
        for n in range(pos.shape[2]):
            if circulation[2,n] == 1 or np.all(circulation[:,n] == 0):
                # already circulating about z or box orbit
                continue

            if sum(circulation[:,n]) > 1:
                logger.warning("Circulation about multiple axes - are you sure "
                               "the orbit has been integrated for long enough?")

            if circulation[0,n] == 1:
                circ = 0
            elif circulation[1,n] == 1:
                circ = 1
            else:
                raise RuntimeError("Should never get here...")

            new_pos[circ,:,n] = pos[2,:,n]
            new_pos[2,:,n] = pos[circ,:,n]

            new_vel[circ,:,n] = vel[2,:,n]
            new_vel[2,:,n] = vel[circ,:,n]

        return self.__class__(pos=new_pos, vel=new_vel, t=self.t, potential=self.potential)

    def plot(self, **kwargs):
        """
        Plot the orbit in all projections. This is a thin wrapper around
        `~gary.dynamics.plot_orbits`.

        .. warning::

            This will currently fail for orbits with fewer than 3 dimensions.

        Parameters
        ----------
        **kwargs
            All keyword arguments are passed to `~gary.dynamics.plot_orbits`.
            See the documentation of that function.

        Returns
        -------
        fig : `~matplotlib.Figure`

        """
        default_kwargs = {
            'marker': None,
            'linestyle': '-',
            'labels': ('$x$ [{}]'.format(self.pos.unit),
                       '$y$ [{}]'.format(self.pos.unit),
                       '$z$ [{}]'.format(self.pos.unit))
        }

        for k,v in default_kwargs.items():
            kwargs[k] = kwargs.get(k, v)

        return plot_orbits(self.pos.value, **kwargs)

def combine(*args):
    """
    Combine the input `Orbit` objects into a single object.

    The `Orbits` must all have the same potential and time array.

    Parameters
    ----------
    *args
        Multiple instances of `Orbit` objects.

    Returns
    -------
    obj : orbit_like
        A single objct with positions and velocities stacked along the last axis.
    """

    ndim = None
    time = None
    pot = None
    pos_unit = None
    vel_unit = None
    cls = None

    all_pos = []
    all_vel = []
    for x in args:
        if ndim is None:
            ndim = x.ndim
            pos_unit = x.pos.unit
            vel_unit = x.vel.unit
            time = x.t
            pot = x.potential
            cls = x.__class__
        else:
            if x.__class__.__name__ != cls.__name__:
                raise ValueError("All objects must have the same class.")

            if x.ndim != ndim:
                raise ValueError("All objects must have the same dimensionality.")

            # TODO: logic here
            # if time is not None or x.t is not None and not np.all(x.t, time):
                raise ValueError("All orbits must have the same time array.")

            if x.potential != pot:
                raise ValueError("All orbits must have the same Potential object.")

        pos = x.pos
        if pos.ndim < 3:
            pos = pos[...,np.newaxis]

        vel = x.vel
        if vel.ndim < 3:
            vel = vel[...,np.newaxis]

        all_pos.append(pos.to(pos_unit).value)
        all_vel.append(vel.to(vel_unit).value)

    all_pos = np.dstack(all_pos)*pos_unit
    all_vel = np.dstack(all_vel)*vel_unit

    return cls(pos=all_pos, vel=all_vel)
