# coding: utf-8

""" General dynamics utilities. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy import log as logger
import astropy.coordinates as coord
import astropy.units as u
uno = u.dimensionless_unscaled
import numpy as np
from scipy.signal import argrelmax, argrelmin

# Project
from ..coordinates import velocity_transforms as vtrans
from ..coordinates import vgal_to_hel
from ..units import UnitSystem
from ..util import atleast_2d

__all__ = ['CartesianPhaseSpacePosition', 'classify_orbit', 'align_circulation_with_z',
           'check_for_primes', 'peak_to_peak_period']

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

    # Pseudo-backwards compatibility
    def w(self, units=None):
        """
        This returns a single array containing the phase-space positions.

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

        if units is None:
            x = self.pos.value
            v = self.vel.value

        else:
            x = self.pos.decompose(units).value
            v = self.vel.decompose(units).value

        return np.vstack((x,v))

    @classmethod
    def from_w(cls, w, units, **kwargs):
        """
        Create a

        Parameters
        ----------
        units : `gary.units.UnitSystem` (optional)
            TODO
        **kwargs
            Any aditional keyword arguments passed to the class initializer.

        Returns
        -------
        TODO...

        """
        units = UnitSystem(units)

        ndim = w.shape[0]//2
        pos = w[:ndim]*units['length']
        vel = w[ndim:]*units['length']/units['time'] # velocity in w is from _core_units

        return cls(pos=pos, vel=vel, **kwargs)

    # ------------------------------------------------------------------------
    # Computed dynamical quantities
    # ------------------------------------------------------------------------
    def kinetic_energy(self):
        """
        The kinetic energy. This is currently *not* cached and is
        computed each time the attribute is accessed.
        """
        return 0.5*np.sum(self.vel**2, axis=0)

    def potential_energy(self, potential):
        """
        The potential energy. This is currently *not* cached and is
        computed each time the attribute is accessed.
        """
        if self.potential is None:
            raise ValueError("To compute the potential energy, a potential"
                             " object must be provided when creating the"
                             " orbit object!")

        # TODO: will I overhaul how potentials handle units?
        q = self.pos.decompose(self.potential.units).value
        _unit = (self.potential.units['length']/self.potential.units['time'])**2
        return self.potential.value(q)*_unit

    def energy(self, potential):
        """
        The total energy (kinetic + potential). This is currently *not*
        cached and is computed each time the attribute is accessed.
        """
        return self.kinetic_energy() + self.potential_energy(potential)

    def angular_momentum(self):
        r"""
        Compute the angular momentum for the phase-space positions contained
        in this object::

        .. math::

            \boldsymbol{L} = \boldsymbol{q} \times \boldsymbol{p}

        See :ref:`shape-conventions` for more information about the shapes of
        input and output objects.

        Returns
        -------
        L : :class:`~astropy.units.Quantity`
            Array of angular momentum vectors.

        Examples
        --------

            >>> import numpy as np
            >>> import astropy.units as u
            >>> pos = np.array([1., 0, 0]) * u.au
            >>> vel = np.array([0, 2*np.pi, 0]) * u.au/u.yr
            >>> orb = CartesianOrbit(pos, vel)
            >>> orb.angular_momentum()
            <Quantity [ 0.        , 0.        , 6.28318531] AU2 / yr>
        """
        return np.cross(self.pos.value, self.vel.value, axis=0) * self.pos.unit * self.vel.unit

    # ------------------------------------------------------------------------
    # Misc. useful methods
    # ------------------------------------------------------------------------
    def plot(self):
        pass

def classify_orbit(w):
    """
    Determine whether an orbit or series of orbits is a Box or Tube orbit by
    figuring out whether there is a change of sign of the angular momentum
    about an axis. Returns a 2D array with 3 integers per orbit point such that:

    - Box and boxlet = [0,0,0]
    - z-axis (short-axis) tube = [0,0,1]
    - x-axis (long-axis) tube = [1,0,0]

    Parameters
    ----------
    w : array_like
        Array of phase-space positions. Accepts 2D or 3D arrays. If 2D, assumes
        this is a single orbit. If 3D, assumes that this is a collection of orbits.
        See :ref:`shape-conventions` for more information about shapes.

    Returns
    -------
    circulation : :class:`numpy.ndarray`
        An array that specifies whether there is circulation about any of
        the axes of the input orbit. For a single orbit, will return a
        1D array, but for multiple orbits, the shape will be ``(3, len(w))``.

    """
    # get angular momenta
    full_ndim = w.shape[0]
    Ls = angular_momentum(w[:full_ndim//2], w[full_ndim//2:])

    # if only 2D, add another empty axis
    if w.ndim == 2:
        single_orbit = True
        Ls = Ls[...,None]
    else:
        single_orbit = False

    ndim,ntimes,norbits = Ls.shape

    # initial angular momentum
    L0 = Ls[:,0]

    # see if at any timestep the sign has changed
    loop = np.ones((ndim,norbits))
    for ii in range(ndim):
        cnd = (np.sign(L0[ii]) != np.sign(Ls[ii,1:])) | \
              (np.abs(Ls[ii,1:]) < 1E-13)
        ix = np.atleast_1d(np.any(cnd, axis=0))
        loop[ii,ix] = 0

    loop = loop.astype(int)
    if single_orbit:
        return loop.reshape((ndim,))
    else:
        return loop

def align_circulation_with_z(w, loop_bit):
    """
    If the input orbit is a tube orbit, this function aligns the circulation
    axis with the z axis.

    Parameters
    ----------
    w : array_like
        Array of phase-space positions. Accepts 2D or 3D arrays. If 2D, assumes
        this is a single orbit. If 3D, assumes that this is a collection of orbits.
        See :ref:`shape-conventions` for more information about shapes.
    loop_bit : array_like
        Array of bits that specify the axis about which the orbit circulates.
        See the documentation for `~gary.dynamics.classify_orbit`.

    Returns
    -------
    new_w : :class:`~numpy.ndarray`
        A copy of the input array with circulation aligned with the z axis.
    """

    if (w.ndim-1) != loop_bit.ndim:
        raise ValueError("Shape mismatch - input orbit array should have 1 more dimension "
                         "than the input loop bit.")

    orig_shape = w.shape
    if loop_bit.ndim == 1:
        loop_bit = atleast_2d(loop_bit,insert_axis=1)
        w = w[...,np.newaxis]
    elif loop_bit.ndim > 2:
        raise ValueError("Invalid shape for loop_bit: {}".format(loop_bit.shape))

    new_w = w.copy()
    for ix in range(w.shape[-1]):
        if loop_bit[2,ix] == 1 or np.all(loop_bit[:,ix] == 0):
            # already circulating about z or box orbit
            continue

        if sum(loop_bit[:,ix]) > 1:
            logger.warning("Circulation about x and y axes - are you sure "
                           "the orbit has been integrated for long enough?")

        if loop_bit[0,ix] == 1:
            circ = 0
        elif loop_bit[1,ix] == 1:
            circ = 1
        else:
            raise RuntimeError("Should never get here...")

        new_w[circ,:,ix] = w[2,:,ix]
        new_w[2,:,ix] = w[circ,:,ix]
        new_w[circ+3,:,ix] = w[5,:,ix]
        new_w[5,:,ix] = w[circ+3,:,ix]

    return new_w.reshape(orig_shape)

def check_for_primes(n, max_prime=41):
    """
    Given an integer, ``n``, ensure that it doest not have large prime
    divisors, which can wreak havok for FFT's. If needed, will decrease
    the number.

    Parameters
    ----------
    n : int
        Integer number to test.

    Returns
    -------
    n2 : int
        Integer combed for large prime divisors.
    """

    m = n
    f = 2
    while (f**2 <= m):
        if m % f == 0:
            m /= f
        else:
            f += 1

    if m >= max_prime and n >= max_prime:
        n -= 1
        n = check_for_primes(n)

    return n

def peak_to_peak_period(t, f, tol=1E-2):
    """
    Estimate the period of the input time series by measuring the average
    peak-to-peak time.

    Parameters
    ----------
    t : array_like
        Time grid aligned with the input time series.
    f : array_like
        A periodic time series.
    tol : numeric (optional)
        A tolerance parameter. Fails if the mean amplitude of oscillations
        isn't larger than this tolerance.

    Returns
    -------
    period : float
        The mean peak-to-peak period.
    """

    # find peaks
    max_ix = argrelmax(f, mode='wrap')[0]
    max_ix = max_ix[(max_ix != 0) & (max_ix != (len(f)-1))]

    # find troughs
    min_ix = argrelmin(f, mode='wrap')[0]
    min_ix = min_ix[(min_ix != 0) & (min_ix != (len(f)-1))]

    # neglect minor oscillations
    if abs(np.mean(f[max_ix]) - np.mean(f[min_ix])) < tol:
        return np.nan

    # compute mean peak-to-peak
    if len(max_ix) > 0:
        T_max = np.mean(t[max_ix[1:]] - t[max_ix[:-1]])
    else:
        T_max = np.nan

    # now compute mean trough-to-trough
    if len(min_ix) > 0:
        T_min = np.mean(t[min_ix[1:]] - t[min_ix[:-1]])
    else:
        T_min = np.nan

    # then take the mean of these two
    return np.mean([T_max, T_min])
