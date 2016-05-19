# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy import log as logger
import astropy.units as u
uno = u.dimensionless_unscaled
import numpy as np
from scipy.signal import argrelmin, argrelmax

# Project
from .core import CartesianPhaseSpacePosition
from .util import peak_to_peak_period
from .plot import plot_orbits
from ..util import atleast_2d

__all__ = ['CartesianOrbit', 'combine']

class Orbit(object):

    # ------------------------------------------------------------------------
    # Shape and size
    # ------------------------------------------------------------------------
    @property
    def ntimes(self):
        return self.pos.shape[1]

    @property
    def norbits(self):
        if self.pos.ndim < 3:
            return 1
        else:
            return self.pos.shape[2]

    # ------------------------------------------------------------------------
    # Computed dynamical quantities
    # ------------------------------------------------------------------------
    def pericenter(self, type=np.mean):
        """
        Estimate the pericenter(s) of the orbit. By default, this returns
        the mean pericenter. To get, e.g., the minimum pericenter,
        pass in ``type=np.min``. To get all pericenters, pass in
        ``type=None``.

        Parameters
        ----------
        type : func (optional)
            By default, this returns the mean pericenter. To return all
            pericenters, pass in ``None``. To get, e.g., the minimum
            or maximum pericenter, pass in ``np.min`` or ``np.max``.

        Returns
        -------
        peri : float, :class:`~numpy.ndarray`
            Either a single number or an array of pericenters.
        """
        r = self.r
        min_ix = argrelmin(r, mode='wrap')[0]
        min_ix = min_ix[(min_ix != 0) & (min_ix != (len(r)-1))]

        if type is not None:
            return type(r[min_ix])
        else:
            return r[min_ix]

    def apocenter(self, type=np.mean):
        """
        Estimate the apocenter(s) of the orbit. By default, this returns
        the mean apocenter. To get, e.g., the minimum apocenter,
        pass in ``type=np.min``. To get all apocenters, pass in
        ``type=None``.

        Parameters
        ----------
        type : func (optional)
            By default, this returns the mean apocenter. To return all
            apocenters, pass in ``None``. To get, e.g., the minimum
            or maximum apocenter, pass in ``np.min`` or ``np.max``.

        Returns
        -------
        apo : float, :class:`~numpy.ndarray`
            Either a single number or an array of apocenters.
        """
        r = self.r
        max_ix = argrelmax(r, mode='wrap')[0]
        max_ix = max_ix[(max_ix != 0) & (max_ix != (len(r)-1))]

        if type is not None:
            return type(r[max_ix])
        else:
            return r[max_ix]

    def eccentricity(self):
        r"""
        Returns the eccentricity computed from the mean apocenter and
        mean pericenter.

        .. math::

            e = \frac{r_{\rm apo} - r_{\rm per}}{r_{\rm apo} + r_{\rm per}}

        Returns
        -------
        ecc : float
            The orbital eccentricity.

        """
        ra = self.apocenter()
        rp = self.pericenter()
        return (ra - rp) / (ra + rp)

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
            r = self.r.value
            if self.norbits == 1:
                T = peak_to_peak_period(self.t.value, r)
                T = T * self.t.unit
            else:
                T = [peak_to_peak_period(self.t.value, r[:,n]) for n in range(r.shape[1])]
                T = T * self.t.unit

        else:
            raise NotImplementedError("sorry 'bout that...")

        return T

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
    potential : `~gala.potential.PotentialBase` (optional)
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

    def __getitem__(self, slyce):
        if isinstance(slyce, np.ndarray) or isinstance(slyce, list):
            _slyce = np.array(slyce)
            _slyce = (slice(None),) + (slyce,)
        else:
            try:
                _slyce = (slice(None),) + tuple(slyce)
            except TypeError:
                _slyce = (slice(None),) + (slyce,)

        kw = dict()
        if self.t is not None:
            kw['t'] = self.t[_slyce[1]]

        pos = self.pos[_slyce]
        vel = self.vel[_slyce]

        if isinstance(_slyce[1], int) or pos.ndim == 1:
            return CartesianPhaseSpacePosition(pos=pos, vel=vel)
        else:
            return self.__class__(pos=pos, vel=vel,
                                  potential=self.potential, **kw)

    def w(self, units=None):
        """
        This returns a single array containing the phase-space positions.

        Parameters
        ----------
        units : `~gala.units.UnitSystem` (optional)
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

    @property
    def r(self):
        """
        The orbital radius.
        """
        return np.sqrt(np.sum(self.pos**2, axis=0))

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
            :meth:`~gala.dynamics.CartesianOrbit.circulation`. See that method for
            more information.

        Returns
        -------
        orb : :class:`~gala.dynamics.CartesianOrbit`
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
        `~gala.dynamics.plot_orbits` -- the docstring for this function is
        included here.

        .. warning::

            This will currently fail for orbits with fewer than 3 dimensions.

        Parameters
        ----------
        ix : int, array_like (optional)
            Index or array of indices of orbits to plot. For example, if `x` is an
            array of shape ``(3,1024,32)`` - 1024 timesteps for 32 orbits in 3D
            positions -- `ix` would specify which of the 32 orbits to plot.
        axes : array_like (optional)
            Array of matplotlib Axes objects.
        triangle : bool (optional)
            Make a triangle plot instead of plotting all projections in a single row.
        subplots_kwargs : dict (optional)
            Dictionary of kwargs passed to :func:`~matplotlib.pyplot.subplots`.
        labels : iterable (optional)
            List or iterable of axis labels as strings. They should correspond to the
            dimensions of the input orbit.
        **kwargs
            All other keyword arguments are passed to :func:`~matplotlib.pyplot.plot`.
            You can pass in any of the usual style kwargs like ``color=...``,
            ``marker=...``, etc.

        Returns
        -------
        fig : `~matplotlib.Figure`

        """
        _label_unit = ''
        if self.pos.unit != u.dimensionless_unscaled:
            _label_unit = ' [{}]'.format(self.pos.unit)

        default_kwargs = {
            'marker': None,
            'linestyle': '-',
            'labels': ('$x${}'.format(_label_unit),
                       '$y${}'.format(_label_unit),
                       '$z${}'.format(_label_unit))
        }

        for k,v in default_kwargs.items():
            kwargs[k] = kwargs.get(k, v)

        return plot_orbits(self.pos.value, **kwargs)

def combine(args, along_time_axis=False):
    """
    Combine the input `Orbit` objects into a single object.

    The `Orbits` must all have the same potential and time array.

    Parameters
    ----------
    args : iterable
        Multiple instances of `Orbit` objects.
    along_time_axis : bool (optional)
        If True, will combine single orbits along the time axis.

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
    all_time = []
    for x in args:
        if ndim is None:
            ndim = x.ndim
            pos_unit = x.pos.unit
            vel_unit = x.vel.unit
            time = x.t
            if time is not None:
                t_unit = time.unit
            else:
                t_unit = None
            pot = x.potential
            cls = x.__class__
        else:
            if x.__class__.__name__ != cls.__name__:
                raise ValueError("All objects must have the same class.")

            if x.ndim != ndim:
                raise ValueError("All objects must have the same dimensionality.")

            if not along_time_axis:
                if time is not None:
                    if x.t is None or len(x.t) != len(time) or np.any(x.t.to(time.unit).value != time.value):
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
        if time is not None:
            all_time.append(x.t.to(t_unit).value)

    norbits = np.array([pos.shape[-1] for pos in all_pos] + [pos.shape[-1] for pos in all_vel])
    if along_time_axis:
        if np.all(norbits == norbits[0]):
            all_pos = np.hstack(all_pos)*pos_unit
            all_vel = np.hstack(all_vel)*vel_unit
            if len(all_time) > 0:
                all_time = np.concatenate(all_time)*t_unit
            else:
                all_time = None
        else:
            raise ValueError("To combine along time axis, all orbit objects must have "
                             "the same number of orbits.")
        if args[0].pos.ndim == 2:
            all_pos = all_pos[...,0]
            all_vel = all_vel[...,0]

    else:
        all_pos = np.dstack(all_pos)*pos_unit
        all_vel = np.dstack(all_vel)*vel_unit
        if len(all_time) > 0:
            all_time = all_time[0]*t_unit
        else:
            all_time = None

    return cls(pos=all_pos, vel=all_vel, t=all_time, potential=args[0].potential)
