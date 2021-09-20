# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from scipy.signal import argrelmax

# Project
from gala.logging import logger
from .core import PhaseSpacePosition
from .util import peak_to_peak_period
from .plot import plot_projections
from ..io import quantity_to_hdf5, quantity_from_hdf5
from ..util import atleast_2d
from ..units import dimensionless, UnitSystem, DimensionlessUnitSystem

__all__ = ['Orbit']


class Orbit(PhaseSpacePosition):
    """
    Represents an orbit: positions and velocities (conjugate momenta) as a
    function of time.

    The class can be instantiated with Astropy representation objects (e.g.,
    :class:`~astropy.coordinates.CartesianRepresentation`), Astropy
    :class:`~astropy.units.Quantity` objects, or plain Numpy arrays.

    If passing in Quantity or Numpy array instances for both position and
    velocity, they are assumed to be Cartesian. Array inputs are interpreted as
    dimensionless quantities. The input position and velocity objects can have
    an arbitrary number of (broadcastable) dimensions. For Quantity or array
    inputs, the first axes have special meaning:

        - ``axis=0`` is the coordinate dimension (e.g., x, y, z)
        - ``axis=1`` is the time dimension

    So if the input position array, ``pos``, has shape ``pos.shape = (3, 100)``,
    this would be a 3D orbit at 100 times (``pos[0]`` is ``x``, ``pos[1]``` is
    ``y``, etc.). For representing multiple orbits, the position array could
    have 3 axes, e.g., it might have shape `pos.shape = (3, 100, 8)`, where this
    is interpreted as a 3D position at 100 times for 8 different orbits. The
    same is true for velocity. The position and velocity arrays must have the
    same shape.

    If a time argument is specified, the position and velocity arrays must have
    the same number of timesteps as the length of the time object::

        len(t) == pos.shape[1]

    Parameters
    ----------
    pos : representation, quantity_like, or array_like
        Positions. If a numpy array (e.g., has no units), this will be
        stored as a dimensionless :class:`~astropy.units.Quantity`. See
        the note above about the assumed meaning of the axes of this object.
    vel : differential, quantity_like, or array_like
        Velocities. If a numpy array (e.g., has no units), this will be
        stored as a dimensionless :class:`~astropy.units.Quantity`. See
        the note above about the assumed meaning of the axes of this object.
    t : array_like, :class:`~astropy.units.Quantity` (optional)
        Array of times. If a numpy array (e.g., has no units), this will be
        stored as a dimensionless :class:`~astropy.units.Quantity`.
    hamiltonian : `~gala.potential.Hamiltonian` (optional)
        The Hamiltonian that the orbit was integrated in.

    """
    def __init__(self, pos, vel, t=None,
                 hamiltonian=None, potential=None, frame=None):

        super().__init__(pos=pos, vel=vel)

        if self.pos.ndim < 1:
            self.pos = self.pos.reshape(1)
            self.vel = self.vel.reshape(1)

        # TODO: check that Hamiltonian ndim is consistent with here

        if t is not None:
            t = np.atleast_1d(t)
            if self.pos.shape[0] != len(t):
                raise ValueError("Position and velocity must have the same "
                                 "length along axis=1 as the length of the "
                                 "time array {} vs {}"
                                 .format(len(t), self.pos.shape[0]))

            if not hasattr(t, 'unit'):
                t = t * u.one

        self.t = t

        if hamiltonian is not None:
            self.potential = hamiltonian.potential
            self.frame = hamiltonian.frame

        else:
            self.potential = potential
            self.frame = frame

    def __getitem__(self, slice_):

        if isinstance(slice_, np.ndarray) or isinstance(slice_, list):
            slice_ = (slice_,)

        try:
            slice_ = tuple(slice_)
        except TypeError:
            slice_ = (slice_,)

        kw = dict()
        if self.t is not None:
            kw['t'] = self.t[slice_[0]]

        pos = self.pos[slice_]
        vel = self.vel[slice_]

        # if one time is sliced out, return a phasespaceposition
        try:
            int_tslice = int(slice_[0])
        except TypeError:
            int_tslice = None

        if int_tslice is not None:
            return PhaseSpacePosition(pos=pos, vel=vel, frame=self.frame)

        else:
            return self.__class__(pos=pos, vel=vel,
                                  potential=self.potential,
                                  frame=self.frame, **kw)

    @property
    def hamiltonian(self):
        if self.potential is None or self.frame is None:
            return None

        try:
            return self._hamiltonian
        except AttributeError:
            from gala.potential import Hamiltonian
            self._hamiltonian = Hamiltonian(potential=self.potential,
                                            frame=self.frame)

        return self._hamiltonian

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
            Will have shape ``(2*ndim, ...)``.

        """

        if units is None:
            if self.hamiltonian is None:
                units = dimensionless
            else:
                units = self.hamiltonian.units

        return super().w(units=units)

    # ------------------------------------------------------------------------
    # Convert from Cartesian to other representations
    #
    def represent_as(self, new_pos, new_vel=None):
        """
        Represent the position and velocity of the orbit in an alternate
        coordinate system. Supports any of the Astropy coordinates
        representation classes.

        Parameters
        ----------
        new_pos : :class:`~astropy.coordinates.BaseRepresentation`
            The type of representation to generate. Must be a class (not an
            instance), or the string name of the representation class.
        new_vel : :class:`~astropy.coordinates.BaseDifferential` (optional)
            Class in which any velocities should be represented. Must be a class
            (not an instance), or the string name of the differential class. If
            None, uses the default differential for the new position class.

        Returns
        -------
        new_orbit : `gala.dynamics.Orbit`
        """
        kw = dict()
        if self.t is not None:
            kw['t'] = self.t
        o = super().represent_as(new_pos=new_pos, new_vel=new_vel)
        return self.__class__(pos=o.pos,
                              vel=o.vel,
                              hamiltonian=self.hamiltonian,
                              **kw)

    # ------------------------------------------------------------------------
    # Shape and size
    # ------------------------------------------------------------------------
    @property
    def ntimes(self):
        return self.shape[0]

    @property
    def norbits(self):
        if len(self.shape) < 2:
            return 1
        else:
            return self.shape[1]

    def reshape(self, new_shape):
        """
        Reshape the underlying position and velocity arrays.
        """
        kw = dict()
        if self.t is not None:
            kw['t'] = self.t
        return self.__class__(pos=self.pos.reshape(new_shape),
                              vel=self.vel.reshape(new_shape),
                              hamiltonian=self.hamiltonian,
                              **kw)

    # ------------------------------------------------------------------------
    # Input / output
    #
    def to_hdf5(self, f):
        """
        Serialize this object to an HDF5 file.

        Requires ``h5py``.

        Parameters
        ----------
        f : str, :class:`h5py.File`
            Either the filename or an open HDF5 file.
        """

        f = super().to_hdf5(f)

        if self.potential is not None:
            import yaml
            from ..potential.potential.io import to_dict
            f['potential'] = yaml.dump(to_dict(self.potential)).encode('utf-8')

        if self.t:
            quantity_to_hdf5(f, 'time', self.t)

        return f

    @classmethod
    def from_hdf5(cls, f):
        """
        Load an object from an HDF5 file.

        Requires ``h5py``.

        Parameters
        ----------
        f : str, :class:`h5py.File`
            Either the filename or an open HDF5 file.
        """
        # TODO: this is duplicated code from PhaseSpacePosition
        if isinstance(f, str):
            import h5py
            f = h5py.File(f, mode='r')
            close = True
        else:
            close = False

        pos = quantity_from_hdf5(f['pos'])
        vel = quantity_from_hdf5(f['vel'])

        time = None
        if 'time' in f:
            time = quantity_from_hdf5(f['time'])

        frame = None
        if 'frame' in f:
            g = f['frame']

            frame_mod = g.attrs['module']
            frame_cls = g.attrs['class']
            frame_units = [u.Unit(x.decode('utf-8')) for x in g['units']]

            if u.dimensionless_unscaled in frame_units:
                units = DimensionlessUnitSystem()
            else:
                units = UnitSystem(*frame_units)

            pars = dict()
            for k in g['parameters']:
                pars[k] = quantity_from_hdf5(g['parameters/'+k])

            exec("from {0} import {1}".format(frame_mod, frame_cls))
            frame_cls = eval(frame_cls)

            frame = frame_cls(units=units, **pars)

        potential = None
        if 'potential' in f:
            import yaml
            from ..potential.potential.io import from_dict
            _dict = yaml.load(f['potential'][()].decode('utf-8'),
                              Loader=yaml.Loader)
            potential = from_dict(_dict)

        if close:
            f.close()

        return cls(pos=pos, vel=vel, t=time,
                   frame=frame, potential=potential)

    def orbit_gen(self):
        """
        Generator for iterating over each orbit.
        """
        if self.norbits == 1:
            yield self

        else:
            for i in range(self.norbits):
                yield self[:, i]

    # ------------------------------------------------------------------------
    # Computed dynamical quantities
    #

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
        if self.hamiltonian is None and potential is None:
            raise ValueError("To compute the potential energy, a potential"
                             " object must be provided!")
        if potential is None:
            potential = self.hamiltonian.potential

        return super().potential_energy(potential)

    def energy(self, hamiltonian=None):
        r"""
        The total energy *per unit mass*:

        Parameters
        ----------
        hamiltonian : `gala.potential.Hamiltonian`, `gala.potential.PotentialBase` instance
            The Hamiltonian object to evaluate the energy. If a potential is
            passed in, this assumes a static reference frame.

        Returns
        -------
        E : :class:`~astropy.units.Quantity`
            The total energy.
        """

        if self.hamiltonian is None and hamiltonian is None:
            raise ValueError("To compute the total energy, a hamiltonian"
                             " object must be provided!")

        if hamiltonian is None:
            hamiltonian = self.hamiltonian
        else:
            from gala.potential import Hamiltonian
            hamiltonian = Hamiltonian(hamiltonian)

        return hamiltonian(self)

    def _max_helper(self, arr, approximate=False):
        """
        Helper function for computing extrema (apocenter, pericenter, z_height)
        and times of extrema.

        Parameters
        ----------
        arr : `numpy.ndarray`
        """
        assert self.norbits == 1
        assert self.t[-1] > self.t[0]  # time must increase

        _ix = argrelmax(arr.value, mode='wrap')[0]
        _ix = _ix[(_ix != 0) & (_ix != (len(arr)-1))]  # remove edges
        t = self.t.value

        approx_arr = arr[_ix]
        approx_t = t[_ix]

        if approximate:
            return approx_arr, approx_t * self.t.unit

        better_times = np.zeros(_ix.shape, dtype=float)
        better_arr = np.zeros(_ix.shape, dtype=float)
        for i, j in enumerate(_ix):
            tvals = t[j-1:j+2]
            rvals = arr[j-1:j+2].value
            coeffs = np.polynomial.polynomial.polyfit(tvals, rvals, 2)
            better_times[i] = (-coeffs[1])/(2*coeffs[2])
            better_arr[i] = ((coeffs[2] * better_times[i]**2)
                             + (coeffs[1] * better_times[i]) + coeffs[0])

        return better_arr * arr.unit, better_times * self.t.unit

    def _max_return_helper(self, vals, times, return_times, reduce):
        if return_times:
            if len(vals) == 1:
                return vals[0], times[0]
            else:
                return vals, times

        elif reduce:
            return u.Quantity(vals).reshape(self.shape[1:])

        else:
            return u.Quantity(vals)

    def pericenter(self, return_times=False, func=np.mean,
                   approximate=False):
        """
        Estimate the pericenter(s) of the orbit by identifying local minima in
        the spherical radius, fitting a parabola around these local minima and
        then solving this parabola to find the pericenter(s).

        By default, this returns the mean of all local minima (pericenters). To
        get, e.g., the minimum pericenter, pass in ``func=np.min``. To get
        all pericenters, pass in ``func=None``.

        Parameters
        ----------
        func : func (optional)
            A function to evaluate on all of the identified pericenter times.
        return_times : bool (optional)
            Also return the pericenter times.
        approximate : bool (optional)
            Compute an approximate pericenter by skipping interpolation.

        Returns
        -------
        peri : float, :class:`~numpy.ndarray`
            Either a single number or an array of pericenters.
        times : :class:`~numpy.ndarray` (optional, see ``return_times``)
            If ``return_times=True``, also returns an array of the pericenter
            times.

        """

        if return_times and func is not None:
            raise ValueError("Cannot return times if reducing pericenters "
                             "using an input function. Pass `func=None` if "
                             "you want to return all individual pericenters "
                             "and times.")

        if func is None:
            reduce = False
            func = lambda x: x  # noqa
        else:
            reduce = True

        # time must increase
        if self.t[-1] < self.t[0]:
            self = self[::-1]

        vals = []
        times = []
        for orbit in self.orbit_gen():
            v, t = orbit._max_helper(-orbit.physicsspherical.r,  # pericenter
                                     approximate=approximate)
            vals.append(func(-v))  # negative for pericenter
            times.append(t)

        return self._max_return_helper(vals, times, return_times, reduce)

    def apocenter(self, return_times=False, func=np.mean,
                  approximate=False):
        """
        Estimate the apocenter(s) of the orbit by identifying local maxima in
        the spherical radius, fitting a parabola around these local maxima and
        then solving this parabola to find the apocenter(s).

        By default, this returns the mean of all local maxima (apocenters). To
        get, e.g., the largest apocenter, pass in ``func=np.max``. To get
        all apocenters, pass in ``func=None``.

        Parameters
        ----------
        func : func (optional)
            A function to evaluate on all of the identified apocenter times.
        return_times : bool (optional)
            Also return the apocenter times.
        approximate : bool (optional)
            Compute an approximate apocenter by skipping interpolation.

        Returns
        -------
        apo : float, :class:`~numpy.ndarray`
            Either a single number or an array of apocenters.
        times : :class:`~numpy.ndarray` (optional, see ``return_times``)
            If ``return_times=True``, also returns an array of the apocenter
            times.

        """

        if return_times and func is not None:
            raise ValueError("Cannot return times if reducing apocenters "
                             "using an input function. Pass `func=None` if "
                             "you want to return all individual apocenters "
                             "and times.")

        if func is None:
            reduce = False
            func = lambda x: x  # noqa
        else:
            reduce = True

        # time must increase
        if self.t[-1] < self.t[0]:
            self = self[::-1]

        vals = []
        times = []
        for orbit in self.orbit_gen():
            v, t = orbit._max_helper(orbit.physicsspherical.r,  # apocenter
                                     approximate=approximate)
            vals.append(func(v))
            times.append(t)

        return self._max_return_helper(vals, times, return_times, reduce)

    def zmax(self, return_times=False, func=np.mean,
             approximate=False):
        """
        Estimate the maximum ``z`` height of the orbit by identifying local
        maxima in the absolute value of the ``z`` position, fitting a parabola
        around these local maxima and then solving this parabola to find the
        maximum ``z`` height.

        By default, this returns the mean of all local maxima. To get, e.g., the
        largest ``z`` excursion, pass in ``func=np.max``. To get all ``z``
        maxima, pass in ``func=None``.

        Parameters
        ----------
        func : func (optional)
            A function to evaluate on all of the identified z maximum times.
        return_times : bool (optional)
            Also return the times of maximum.
        approximate : bool (optional)
            Compute approximate values by skipping interpolation.

        Returns
        -------
        zs : float, :class:`~numpy.ndarray`
            Either a single number or an array of maximum z heights.
        times : :class:`~numpy.ndarray` (optional, see ``return_times``)
            If ``return_times=True``, also returns an array of the apocenter
            times.

        """

        if return_times and func is not None:
            raise ValueError("Cannot return times if reducing "
                             "using an input function. Pass `func=None` if "
                             "you want to return all individual values "
                             "and times.")

        if func is None:
            reduce = False
            func = lambda x: x  # noqa
        else:
            reduce = True

        # time must increase
        if self.t[-1] < self.t[0]:
            self = self[::-1]

        vals = []
        times = []
        for orbit in self.orbit_gen():
            v, t = orbit._max_helper(np.abs(orbit.cylindrical.z),
                                     approximate=approximate)
            vals.append(func(v))
            times.append(t)

        return self._max_return_helper(vals, times, return_times, reduce)

    def eccentricity(self, **kw):
        r"""
        Returns the eccentricity computed from the mean apocenter and
        mean pericenter.

        .. math::

            e = \frac{r_{\rm apo} - r_{\rm per}}{r_{\rm apo} + r_{\rm per}}

        Parameters
        ----------
        **kw
            Any keyword arguments passed to ``apocenter()`` and
            ``pericenter()``. For example, ``approximate=True``.

        Returns
        -------
        ecc : float
            The orbital eccentricity.

        """
        ra = self.apocenter(**kw)
        rp = self.pericenter(**kw)
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
            raise ValueError("To compute the period, a time array is needed. "
                             "Specify a time array when creating this object.")

        if radial:
            r = self.physicsspherical.r.value
            if self.norbits == 1:
                T = u.Quantity(peak_to_peak_period(self.t, r))
            else:
                T = u.Quantity([peak_to_peak_period(self.t, r[:, n])
                                for n in range(r.shape[1])])

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

        - Box and boxlet = [0, 0, 0]
        - z-axis (short-axis) tube = [0, 0, 1]
        - x-axis (long-axis) tube = [1, 0, 0]

        Returns
        -------
        circulation : :class:`numpy.ndarray`
            An array that specifies whether there is circulation about any of
            the axes of the input orbit. For a single orbit, will return a
            1D array, but for multiple orbits, the shape will be
            ``(3, norbits)``.

        """
        L = self.angular_momentum()

        # if only 2D, add another empty axis
        if L.ndim == 2:
            single_orbit = True
            L = L[..., None]
        else:
            single_orbit = False

        ndim, ntimes, norbits = L.shape

        # initial angular momentum
        L0 = L[:, 0]

        # see if at any timestep the sign has changed
        circ = np.ones((ndim, norbits))
        for ii in range(ndim):
            cnd = (np.sign(L0[ii]) != np.sign(L[ii, 1:])) | \
                  (np.abs(L[ii, 1:]).value < 1E-13)
            ix = np.atleast_1d(np.any(cnd, axis=0))
            circ[ii, ix] = 0

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
            Array of bits that specify the axis about which the orbit
            circulates. If not provided, will compute this using
            :meth:`~gala.dynamics.Orbit.circulation`. See that method for more
            information.

        Returns
        -------
        orb : :class:`~gala.dynamics.Orbit`
            A copy of the original orbit object with circulation aligned with
            the z axis.
        """

        if circulation is None:
            circulation = self.circulation()
        circulation = atleast_2d(circulation, insert_axis=1)

        cart = self.cartesian
        pos = cart.xyz
        vel = np.vstack((cart.v_x.value[None],
                         cart.v_y.value[None],
                         cart.v_z.value[None])) * cart.v_x.unit

        if pos.ndim < 3:
            pos = pos[..., np.newaxis]
            vel = vel[..., np.newaxis]

        if (circulation.shape[0] != self.ndim or
                circulation.shape[1] != pos.shape[2]):
            raise ValueError("Shape of 'circulation' array should match the "
                             "shape of the position/velocity (minus the time "
                             "axis).")

        new_pos = pos.copy()
        new_vel = vel.copy()
        for n in range(pos.shape[2]):
            if circulation[2, n] == 1 or np.all(circulation[:, n] == 0):
                # already circulating about z or box orbit
                continue

            if sum(circulation[:, n]) > 1:
                logger.warning("Circulation about multiple axes - are you sure "
                               "the orbit has been integrated for long enough?")

            if circulation[0, n] == 1:
                circ = 0
            elif circulation[1, n] == 1:
                circ = 1
            else:
                raise RuntimeError("Should never get here...")

            new_pos[circ, :, n] = pos[2, :, n]
            new_pos[2, :, n] = pos[circ, :, n]

            new_vel[circ, :, n] = vel[2, :, n]
            new_vel[2, :, n] = vel[circ, :, n]

        return self.__class__(pos=new_pos.reshape(cart.xyz.shape),
                              vel=new_vel.reshape(cart.xyz.shape),
                              t=self.t,
                              hamiltonian=self.hamiltonian)

    def plot(self, components=None, units=None, auto_aspect=True, **kwargs):
        """
        Plot the positions in all projections. This is a wrapper around
        `~gala.dynamics.plot_projections` for fast access and quick
        visualization. All extra keyword arguments are passed to that function
        (the docstring for this function is included here for convenience).

        Parameters
        ----------
        components : iterable (optional)
            A list of component names (strings) to plot. By default, this is the
            Cartesian positions ``['x', 'y', 'z']``. To plot Cartesian
            velocities, pass in the velocity component names
            ``['v_x', 'v_y', 'v_z']``. If the representation is different, the
            component names will be different. For example, for a Cylindrical
            representation, the components are ``['rho', 'phi', 'z']`` and
            ``['v_rho', 'pm_phi', 'v_z']``.
        units : `~astropy.units.UnitBase`, iterable (optional)
            A single unit or list of units to display the components in.
        auto_aspect : bool (optional)
            Automatically enforce an equal aspect ratio.
        relative_to : bool (optional)
            Plot the values relative to this value or values.
        autolim : bool (optional)
            Automatically set the plot limits to be something sensible.
        axes : array_like (optional)
            Array of matplotlib Axes objects.
        subplots_kwargs : dict (optional)
            Dictionary of kwargs passed to :func:`~matplotlib.pyplot.subplots`.
        labels : iterable (optional)
            List or iterable of axis labels as strings. They should correspond to
            the dimensions of the input orbit.
        plot_function : callable (optional)
            The ``matplotlib`` plot function to use. By default, this is
            :func:`~matplotlib.pyplot.scatter`, but can also be, e.g.,
            :func:`~matplotlib.pyplot.plot`.
        **kwargs
            All other keyword arguments are passed to the ``plot_function``.
            You can pass in any of the usual style kwargs like ``color=...``,
            ``marker=...``, etc.

        Returns
        -------
        fig : `~matplotlib.Figure`

        """
        from gala.tests.optional_deps import HAS_MATPLOTLIB
        if not HAS_MATPLOTLIB:
            raise ImportError('matplotlib is required for visualization.')
        import matplotlib.pyplot as plt

        if components is None:
            if self.ndim == 1:  # only a 1D orbit, so just plot time series
                components = ['t', self.pos.components[0]]
            else:
                components = self.pos.components

        x, labels = self._plot_prepare(components=components,
                                       units=units)

        kwargs.setdefault('marker', '')
        kwargs.setdefault('linestyle', '-')
        kwargs.setdefault('labels', labels)
        kwargs.setdefault('plot_function', plt.plot)

        fig = plot_projections(x, **kwargs)

        if self.pos.get_name() == 'cartesian' and \
                all([not c.startswith('d_') for c in components]) and \
                't' not in components and \
                auto_aspect:
            for ax in fig.axes:
                ax.set(aspect='equal', adjustable='datalim')

        return fig

    def plot_3d(self, components=None, units=None, auto_aspect=True,
                subplots_kwargs=None, **kwargs):
        """
        Plot the specified 3D components.

        Parameters
        ----------
        components : iterable (optional)
            A list of component names (strings) to plot. By default, this is the
            Cartesian positions ``['x', 'y', 'z']``. To plot Cartesian
            velocities, pass in the velocity component names
            ``['v_x', 'v_y', 'v_z']``. If the representation is different, the
            component names will be different. For example, for a Cylindrical
            representation, the components are ``['rho', 'phi', 'z']`` and
            ``['v_rho', 'pm_phi', 'v_z']``.
        units : `~astropy.units.UnitBase`, iterable (optional)
            A single unit or list of units to display the components in.
        auto_aspect : bool (optional)
            Automatically enforce an equal aspect ratio.
        ax : `matplotlib.axes.Axes`
            The matplotlib Axes object to draw on.
        subplots_kwargs : dict (optional)
            Dictionary of kwargs passed to :func:`~matplotlib.pyplot.subplots`.
        labels : iterable (optional)
            List or iterable of axis labels as strings. They should correspond
            to the dimensions of the input orbit.
        plot_function : str (optional)
            The ``matplotlib`` plot function to use. By default, this is 'plot'
            but can also be, e.g., 'scatter'.
        **kwargs
            All other keyword arguments are passed to the ``plot_function``.
            You can pass in any of the usual style kwargs like ``color=...``,
            ``marker=...``, etc.

        Returns
        -------
        fig : `~matplotlib.Figure`

        """
        from gala.tests.optional_deps import HAS_MATPLOTLIB
        if not HAS_MATPLOTLIB:
            raise ImportError('matplotlib is required for visualization.')
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d  # noqa

        if components is None:
            components = self.pos.components

        if subplots_kwargs is None:
            subplots_kwargs = dict()

        if len(components) != 3:
            raise ValueError(
                f"The number of components ({len(components)}) must be 3")

        x, labels = self._plot_prepare(components=components,
                                       units=units)

        kwargs.setdefault('marker', '')
        kwargs.setdefault('linestyle', kwargs.pop('ls', '-'))
        plot_function_name = kwargs.pop('plot_function', 'plot')

        ax = kwargs.pop('ax', None)
        subplots_kwargs.setdefault('constrained_layout', True)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6),
                                   subplot_kw=dict(projection='3d'),
                                   **subplots_kwargs)
        else:
            fig = ax.figure

        plot_function = getattr(ax, plot_function_name)
        if x[0].ndim > 1:
            for n in range(x[0].shape[1]):
                plot_function(*[xx[:, n] for xx in x], **kwargs)
        else:
            plot_function(*x, **kwargs)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])

        if self.pos.get_name() == 'cartesian' and \
                all([not c.startswith('d_') for c in components]) and \
                't' not in components and \
                auto_aspect:
            for ax in fig.axes:
                ax.set(aspect='auto', adjustable='datalim')

        return fig, ax

    def animate(self,
                components=None,
                units=None,
                stride=1,
                segment_nsteps=10,
                underplot_full_orbit=True,
                marker_style=None,
                segment_style=None,
                FuncAnimation_kwargs=None,
                orbit_plot_kwargs=None,
                axes=None):
        """
        Animate an orbit or collection of orbits.

        Parameters
        ----------
        components : iterable (optional)
            A list of component names (strings) to plot. By default, this is the
            Cartesian positions ``['x', 'y', 'z']``. To plot Cartesian
            velocities, pass in the velocity component names
            ``['v_x', 'v_y', 'v_z']``. If the representation is different, the
            component names will be different. For example, for a Cylindrical
            representation, the components are ``['rho', 'phi', 'z']`` and
            ``['v_rho', 'pm_phi', 'v_z']``.
        units : `~astropy.units.UnitBase`, iterable (optional)
            A single unit or list of units to display the components in.
        stride : int (optional)
            How often to draw a new frame, in terms of orbit timesteps.
        segment_nsteps : int (optional)
            How many timesteps to draw in an orbit segment trailing
            the timestep marker. Set this to 0 or None to disable.
        underplot_full_orbit : bool (optional)
            Controls whether to under-plot the full orbit as a thin line.
        marker_style : dict (optional)
            Matplotlib style arguments passed to `matplotlib.pyplot.plot`
            that control the plot style of the timestep marker.
        segment_style : dict (optional)
            Matplotlib style arguments passed to `matplotlib.pyplot.plot`
            that control the plot style of the orbit segment.
        FuncAnimation_kwargs : dict (optional)
            Keyword arguments passed through to
            `matplotlib.animation.FuncAnimation`.
        orbit_plot_kwargs : dict (optional)
            Keyword arguments passed through to `gala.dynamics.Orbit.plot`.
        axes : `matplotlib.axes.Axes` (optional)
            Where to draw the orbit.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
        anim : `matplotlib.animation.FuncAnimation`

        """
        from gala.tests.optional_deps import HAS_MATPLOTLIB
        if not HAS_MATPLOTLIB:
            raise ImportError('matplotlib is required for visualization.')
        from matplotlib.animation import FuncAnimation

        if components is None:
            if self.ndim == 1:  # only a 1D orbit, so just plot time series
                components = ['t', self.pos.components[0]]
            else:
                components = self.pos.components

        # Extract the relevant components, in the given unit system
        xs, _ = self._plot_prepare(components=components,
                                   units=units)
        xs = [atleast_2d(xx, insert_axis=1) for xx in xs]

        # Figure out which components to plot on which axes
        data_paired = []
        for i in range(len(xs)):
            for j in range(len(xs)):
                if i >= j:
                    continue  # skip diagonal, upper triangle
                data_paired.append((xs[i], xs[j]))

        if FuncAnimation_kwargs is None:
            FuncAnimation_kwargs = dict()

        if orbit_plot_kwargs is None:
            orbit_plot_kwargs = dict()
        orbit_plot_kwargs.setdefault('zorder', 1)
        orbit_plot_kwargs.setdefault('color', '#aaaaaa')
        orbit_plot_kwargs.setdefault('linewidth', '1')
        orbit_plot_kwargs.setdefault('axes', axes)

        if marker_style is None:
            marker_style = dict()
        marker_style.setdefault('marker', 'o')
        marker_style.setdefault('linestyle', marker_style.pop('ls', 'None'))
        marker_style.setdefault('markersize', marker_style.pop('ms', 4.))
        marker_style.setdefault('color', marker_style.pop('c', 'tab:red'))
        marker_style.setdefault('zorder', 100)

        if segment_style is None:
            segment_style = dict()
        segment_style.setdefault('marker', 'None')
        segment_style.setdefault('linestyle', segment_style.pop('ls', '-'))
        segment_style.setdefault('linewidth', segment_style.pop('lw', 2.))
        segment_style.setdefault('color', segment_style.pop('c', 'tab:blue'))
        segment_style.setdefault('zorder', 10)
        if segment_nsteps is None or segment_nsteps == 0:  # HACK
            segment_style['alpha'] = 0

        # Use this to get a figure with axes with the right limits
        # Note: Labels are added by .plot()
        if not underplot_full_orbit:
            orbit_plot_kwargs['alpha'] = 0
        fig = self.plot(components=components, units=units,
                        **orbit_plot_kwargs)

        # Set up all of the (data-less) markers and line segments
        markers = []
        segments = []
        for n in range(self.norbits):
            _m = []
            _s = []
            for i in range(len(data_paired)):
                _m.append(fig.axes[i].plot([], [], **marker_style)[0])
                _s.append(fig.axes[i].plot([], [], **segment_style)[0])
            markers.append(_m)
            segments.append(_s)

        def anim_func(n):
            i = max(0, n - segment_nsteps)

            for k in range(self.norbits):
                for j in range(len(data_paired)):
                    markers[k][j].set_data(data_paired[j][0][n:n+1, k],
                                           data_paired[j][1][n:n+1, k])
                    segments[k][j].set_data(data_paired[j][0][i:n+1, k],
                                            data_paired[j][1][i:n+1, k])

            return (*[m for m in markers for x in m],
                    *[s for s in segments for x in s])

        anim = FuncAnimation(fig, anim_func,
                             frames=np.arange(0, self.ntimes, stride),
                             **FuncAnimation_kwargs)

        return fig, anim

    def to_frame(self, frame, current_frame=None, **kwargs):
        """
        Transform to a different reference frame.

        Parameters
        ----------
        frame : `gala.potential.CFrameBase`
            The frame to transform to.
        current_frame : `gala.potential.CFrameBase` (optional)
            If the Orbit has no associated Hamiltonian, this specifies the
            current frame of the orbit.

        Returns
        -------
        orbit : `gala.dynamics.Orbit`
            The orbit in the new reference frame.

        """

        kw = kwargs.copy()

        # TODO: this short-circuit sux
        if current_frame is None:
            current_frame = self.frame
        if frame == current_frame and not kwargs:
            return self

        # TODO: need a better way to do this!
        from ..potential.frame.builtin import ConstantRotatingFrame
        for fr in [frame, current_frame, self.frame]:
            if isinstance(fr, ConstantRotatingFrame):
                if 't' not in kw:
                    kw['t'] = self.t

        # TODO: this needs a re-write...
        psp = super().to_frame(frame, current_frame, **kw)

        return Orbit(pos=psp.pos, vel=psp.vel, t=self.t,
                     frame=frame, potential=self.potential)

    # ------------------------------------------------------------------------
    # Compatibility with other packages
    #

    def to_galpy_orbit(self, ro=None, vo=None):
        """Convert this object to a ``galpy.Orbit`` instance.

        Parameters
        ----------
        ro : `astropy.units.Quantity` or `astropy.units.UnitBase`
            "Natural" length unit.
        vo : `astropy.units.Quantity` or `astropy.units.UnitBase`
            "Natural" velocity unit.

        Returns
        -------
        galpy_orbit : `galpy.orbit.Orbit`

        """
        from galpy.orbit import Orbit
        from galpy.util.config import __config__ as galpy_config

        if self.frame is not None:
            from ..potential import StaticFrame
            w = self.to_frame(StaticFrame(self.frame.units))
        else:
            w = self

        if ro is None:
            ro = galpy_config.getfloat('normalization', 'ro')
            ro = ro * u.kpc

        if vo is None:
            vo = galpy_config.getfloat('normalization', 'vo')
            vo = vo * u.km/u.s

        # PhaseSpacePosition or Orbit:
        cyl = w.cylindrical

        R = cyl.rho.to_value(ro).T
        phi = cyl.phi.to_value(u.rad).T
        z = cyl.z.to_value(ro).T

        vR = cyl.v_rho.to_value(vo).T
        vT = (cyl.rho * cyl.pm_phi).to_value(vo, u.dimensionless_angles()).T
        vz = cyl.v_z.to_value(vo).T

        o = Orbit(np.array([R, vR, vT, z, vz, phi]).T, ro=ro, vo=vo)
        if w.t is not None:
            o.t = w.t.to_value(ro / vo)

        return o

    @classmethod
    def from_galpy_orbit(self, galpy_orbit):
        """Create a Gala ``PhaseSpacePosition`` or ``Orbit`` instance from a
        ``galpy.Orbit`` instance.

        Parameters
        ----------
        galpy_orbit : :class:`galpy.orbit.Orbit`

        Returns
        -------
        orbit : :class:`~gala.dynamics.Orbit`

        """
        ro = galpy_orbit._ro * u.kpc
        vo = galpy_orbit._vo * u.km/u.s
        ts = galpy_orbit.t

        rep = coord.CylindricalRepresentation(
            rho=galpy_orbit.R(ts) * ro,
            phi=galpy_orbit.phi(ts) * u.rad,
            z=galpy_orbit.z(ts) * ro
        )
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            dif = coord.CylindricalDifferential(
                d_rho=galpy_orbit.vR(ts) * vo,
                d_phi=galpy_orbit.vT(ts) * vo / rep.rho,
                d_z=galpy_orbit.vz(ts) * vo
            )

        t = galpy_orbit.t * ro / vo
        return Orbit(rep, dif, t=t)
