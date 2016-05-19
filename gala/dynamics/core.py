# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy import log as logger
import astropy.coordinates as coord
import astropy.units as u
uno = u.dimensionless_unscaled
import numpy as np

# Project
from .plot import three_panel
from ..coordinates import velocity_transforms as vtrans
from ..coordinates import vgal_to_hel
from ..units import UnitSystem, DimensionlessUnitSystem
from ..util import atleast_2d

__all__ = ['CartesianPhaseSpacePosition', 'combine']

class PhaseSpacePosition(object):

    # ------------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------------
    def __repr__(self):
        return "<{} N={}, shape={}>".format(self.__class__.__name__, self.ndim, self.shape)

    def __str__(self):
        # TODO: should show some arrays?
        return "{} {}D, {}".format(self.__class__.__name__, self.ndim, self.shape)

    # ------------------------------------------------------------------------
    # Shape and size
    # ------------------------------------------------------------------------
    @property
    def ndim(self):
        """
        Number of coordinate dimensions. 1/2 of the phase-space dimensionality.

        .. warning::

            This is *not* the number of axes in the position or velocity
            arrays. That is accessed by doing ``{}.pos.ndim``.

        Returns
        -------
        n : int

        """.format(self.__class__.__name__)
        return self.pos.shape[0]

    @property
    def shape(self):
        """
        .. warning::

            This is *not* the shape of the position or velocity
            arrays. That is accessed by doing ``{}.pos.shape``.

        Returns
        -------
        shp : tuple

        """.format(self.__class__.__name__)
        return self.pos.shape[1:]

class CartesianPhaseSpacePosition(PhaseSpacePosition):
    """
    Represents phase-space positions in Cartesian coordinates, e.g.,
    positions and conjugate momenta (velocities).

    .. warning::

        This is an experimental class. The API may change in a future release!

    The position and velocity quantities (arrays) can have an arbitrary
    number of dimensions, but the first axis (0, 1) has special meaning::

        - `axis=0` is the coordinate dimension (e.g., x, y, z)

    So if the input position array, `pos`, has shape `pos.shape = (3, 100)`,
    this would represent 100 3D positions (`pos[0]` is `x`, `pos[1]` is `y`,
    etc.). The same is true for velocity. The position and velocity arrays
    must have the same shape.

    If the input position and velocity are arrays rather than
    :class:`~astropy.units.Quantity` objects, they are internally stored with
    dimensionles units.

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

    """
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

    def __getitem__(self, slyce):
        if isinstance(slyce, np.ndarray) or isinstance(slyce, list):
            _slyce = np.array(slyce)
            _slyce = (slice(None),) + (slyce,)
        else:
            try:
                _slyce = (slice(None),) + tuple(slyce)
            except TypeError:
                _slyce = (slice(None),) + (slyce,)

        return self.__class__(pos=self.pos[_slyce], vel=self.vel[_slyce])

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
        if self.ndim != 3:
            raise ValueError("Representation changes require a 3D (ndim=3) "
                             "position and velocity.")

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
            Circular velocity of the Sun. Passed to velocity transformation.
        vlsr : :class:`~astropy.units.Quantity`
            Velocity of the Sun relative to the LSR. Passed to
            velocity transformation.

        Returns
        -------
        c : :class:`~astropy.coordinates.BaseCoordinateFrame`
            An instantiated coordinate frame.
        v : tuple
            A tuple of velocities represented as
            :class:`~astropy.units.Quantity` objects.

        """

        if self.ndim != 3:
            raise ValueError("Frame transformations require a 3D (ndim=3) "
                             "position and velocity.")

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
        units : `~gala.units.UnitSystem` (optional)
            The unit system to represent the position and velocity in
            before combining into the full array.

        Returns
        -------
        w : `~numpy.ndarray`
            A numpy array of all positions and velocities, without units.
            Will have shape ``(2*ndim,...)``.

        """
        if (units is None or isinstance(units, DimensionlessUnitSystem)) \
            and (self.pos.unit == uno and self.vel.unit == uno):
            units = DimensionlessUnitSystem()

        elif units is None:
            raise ValueError("A UnitSystem must be provided.")

        x = self.pos.decompose(units).value
        v = self.vel.decompose(units).value

        return np.vstack((x,v))

    @classmethod
    def from_w(cls, w, units=None, **kwargs):
        """
        Create a {name} object from a single array specifying positions
        and velocities. This is mainly for backwards-compatibility and
        it is not recommended for new users.

        Parameters
        ----------
        w : array_like
            The array of phase-space positions.
        units : `~gala.units.UnitSystem` (optional)
            The unit system that the input position+velocity array, ``w``,
            is represented in.
        **kwargs
            Any aditional keyword arguments passed to the class initializer.

        Returns
        -------
        obj : `~gala.dynamics.{name}`

        """.format(name=cls.__name__)

        ndim = w.shape[0]//2
        pos = w[:ndim]
        vel = w[ndim:]

        # TODO: this is bad form - UnitSystem should know what to do with a Dimensionless
        if units is not None and not isinstance(units, DimensionlessUnitSystem):
            units = UnitSystem(units)
            pos = pos*units['length']
            vel = vel*units['length']/units['time'] # velocity in w is from _core_units

        return cls(pos=pos, vel=vel, **kwargs)

    # ------------------------------------------------------------------------
    # Computed dynamical quantities
    # ------------------------------------------------------------------------
    def kinetic_energy(self):
        r"""
        The kinetic energy *per unit mass*:

        .. math::

            E_K = \frac{1}{2} \, |\boldsymbol{v}|^2

        Returns
        -------
        E : :class:`~astropy.units.Quantity`
            The kinetic energy.
        """
        return 0.5*np.sum(self.vel**2, axis=0)

    def potential_energy(self, potential):
        r"""
        The potential energy *per unit mass*:

        .. math::

            E_\Phi = \Phi(\boldsymbol{q})

        Parameters
        ----------
        potential : `gala.potential.PotentialBase`
            The potential object to compute the energy from.

        Returns
        -------
        E : :class:`~astropy.units.Quantity`
            The potential energy.
        """
        return potential.value(self.pos)

    def energy(self, potential):
        r"""
        The total energy *per unit mass* (e.g., kinetic + potential):

        Parameters
        ----------
        potential : `gala.potential.PotentialBase`
            The potential object to compute the energy from.

        Returns
        -------
        E : :class:`~astropy.units.Quantity`
            The total energy.
        """
        return self.kinetic_energy() + self.potential_energy(potential)

    def angular_momentum(self):
        r"""
        Compute the angular momentum for the phase-space positions contained
        in this object::

        .. math::

            \boldsymbol{{L}} = \boldsymbol{{q}} \times \boldsymbol{{p}}

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
            >>> w = {}(pos, vel)
            >>> w.angular_momentum()
            <Quantity [ 0.        , 0.        , 6.28318531] AU2 / yr>
        """.format(self.__class__.__name__)
        return np.cross(self.pos.value, self.vel.value, axis=0) * self.pos.unit * self.vel.unit

    # ------------------------------------------------------------------------
    # Misc. useful methods
    # ------------------------------------------------------------------------
    def plot(self, **kwargs):
        """
        Plot the positions in all projections. This is a thin wrapper around
        `~gala.dynamics.three_panel` -- the docstring for this function is
        included here.

        .. warning::

            This will currently fail for orbits with fewer than 3 dimensions.

        Parameters
        ----------
        relative_to : bool (optional)
            Plot the values relative to this value or values.
        autolim : bool (optional)
            Automatically set the plot limits to be something sensible.
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
            All other keyword arguments are passed to :func:`~matplotlib.pyplot.scatter`.
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
            'marker': '.',
            'color': 'k',
            'labels': ('$x${}'.format(_label_unit),
                       '$y${}'.format(_label_unit),
                       '$z${}'.format(_label_unit))
        }

        for k,v in default_kwargs.items():
            kwargs[k] = kwargs.get(k, v)

        return three_panel(self.pos.value, **kwargs)

def combine(args):
    """
    Combine the input `PhaseSpacePosition` objects into a single object.

    Parameters
    ----------
    args : iterable
        Multiple instances of `PhaseSpacePosition`.

    Returns
    -------
    obj : `~gala.dynamics.PhaseSpacePosition`
        A single objct with positions and velocities stacked along the last axis.
    """

    ndim = None
    pos_unit = None
    vel_unit = None
    all_pos = []
    all_vel = []
    for x in args:
        if ndim is None:
            ndim = x.ndim
            pos_unit = x.pos.unit
            vel_unit = x.vel.unit
        else:
            if x.ndim != ndim:
                raise ValueError("All objects must have the same dimensionality.")

        pos = x.pos
        if pos.ndim < 2:
            pos = pos[...,np.newaxis]

        vel = x.vel
        if vel.ndim < 2:
            vel = vel[...,np.newaxis]

        all_pos.append(pos.to(pos_unit).value)
        all_vel.append(vel.to(vel_unit).value)

    all_pos = np.hstack(all_pos)*pos_unit
    all_vel = np.hstack(all_vel)*vel_unit

    return CartesianPhaseSpacePosition(pos=all_pos, vel=all_vel)

