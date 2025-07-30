import importlib
import re
from collections import namedtuple

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.coordinates import representation as r

from ..io import quantity_from_hdf5, quantity_to_hdf5
from ..units import DimensionlessUnitSystem, UnitSystem, _greek_letters
from ..util import atleast_2d
from . import representation_nd as rep_nd
from .plot import plot_projections

__all__ = ["PhaseSpacePosition"]

_RepresentationMappingBase = namedtuple(
    "RepresentationMapping", ("repr_name", "new_name", "default_unit")
)


class RepresentationMapping(_RepresentationMappingBase):
    """
    This `~collections.namedtuple` is used to override the representation and
    differential class component names in the `PhaseSpacePosition` and `Orbit`
    classes.
    """

    def __new__(cls, repr_name, new_name, default_unit="recommended"):
        # this trick just provides some defaults
        return super().__new__(cls, repr_name, new_name, default_unit)


class RegexRepresentationMapping(RepresentationMapping):
    """
    A representation mapping that uses a regex to map the original attribute
    name to the new attribute name.
    """


class PhaseSpacePosition:
    representation_mappings = {
        r.CartesianRepresentation: [RepresentationMapping("xyz", "xyz")],
        r.SphericalCosLatDifferential: [
            RepresentationMapping("d_lon_coslat", "pm_lon_coslat", u.mas / u.yr),
            RepresentationMapping("d_lat", "pm_lat", u.mas / u.yr),
            RepresentationMapping("d_distance", "radial_velocity"),
        ],
        r.SphericalDifferential: [
            RepresentationMapping("d_lon", "pm_lon", u.mas / u.yr),
            RepresentationMapping("d_lat", "pm_lat", u.mas / u.yr),
            RepresentationMapping("d_distance", "radial_velocity"),
        ],
        r.PhysicsSphericalDifferential: [
            RepresentationMapping("d_phi", "pm_phi", u.mas / u.yr),
            RepresentationMapping("d_theta", "pm_theta", u.mas / u.yr),
            RepresentationMapping("d_r", "radial_velocity"),
        ],
        r.CartesianDifferential: [
            RepresentationMapping("d_x", "v_x"),
            RepresentationMapping("d_y", "v_y"),
            RepresentationMapping("d_z", "v_z"),
            RepresentationMapping("d_xyz", "v_xyz"),
        ],
        r.CylindricalDifferential: [
            RepresentationMapping("d_rho", "v_rho"),
            RepresentationMapping("d_phi", "pm_phi"),
            RepresentationMapping("d_z", "v_z"),
        ],
        rep_nd.NDCartesianRepresentation: [RepresentationMapping("xyz", "xyz")],
        rep_nd.NDCartesianDifferential: [
            RepresentationMapping("d_xyz", "v_xyz"),
            RegexRepresentationMapping("d_x([0-9])", "v_x{0}"),
        ],
    }
    representation_mappings[r.UnitSphericalCosLatDifferential] = (
        representation_mappings[r.SphericalCosLatDifferential]
    )
    representation_mappings[r.UnitSphericalDifferential] = representation_mappings[
        r.SphericalDifferential
    ]

    def __init__(self, pos, vel=None, frame=None):
        """
        Represents phase-space positions, i.e. positions and conjugate momenta
        (velocities).

        The class can be instantiated with Astropy representation objects (e.g.,
        :class:`~astropy.coordinates.CartesianRepresentation`), Astropy
        :class:`~astropy.units.Quantity` objects, or plain Numpy arrays.

        If passing in representation objects, the default representation is
        taken to be the class that is passed in.

        If passing in Quantity or Numpy array instances for both position and
        velocity, they are assumed to be Cartesian. Array inputs are interpreted
        as dimensionless quantities. The input position and velocity objects can
        have an arbitrary number of (broadcastable) dimensions. For Quantity or
        array inputs, the first axis (0) has special meaning:

            - `axis=0` is the coordinate dimension (e.g., x, y, z for Cartesian)

        So if the input position array, `pos`, has shape `pos.shape = (3, 100)`,
        this would represent 100 3D positions (`pos[0]` is `x`, `pos[1]` is `y`,
        etc.). The same is true for velocity.

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
        frame : :class:`~gala.potential.FrameBase` (optional)
            The reference frame of the input phase-space positions.

        """

        if isinstance(pos, coord.Galactocentric):
            pos = pos.data

        if not isinstance(pos, coord.BaseRepresentation):
            # assume Cartesian if not specified
            if not hasattr(pos, "unit"):
                pos *= u.one

            # 3D coordinates get special treatment
            ndim = pos.shape[0]
            if ndim == 3:
                # TODO: HACK: until this stuff is in astropy core
                if isinstance(pos, coord.BaseRepresentation):
                    kw = [(k, getattr(pos, k)) for k in pos.components]
                    pos = getattr(coord, pos.__class__.__name__)(**kw)

                else:
                    pos = coord.CartesianRepresentation(pos)

            else:
                pos = rep_nd.NDCartesianRepresentation(pos)

        else:
            ndim = 3

        if vel is None:
            if "s" not in pos.differentials:
                msg = (
                    "You must specify velocity data when creating "
                    f"a {self.__class__.__name__} object."
                )
                raise TypeError(msg)
            vel = pos.differentials.get("s", None)

        if not isinstance(vel, coord.BaseDifferential):
            # assume representation is same as pos if not specified
            if not hasattr(vel, "unit"):
                vel *= u.one

            if ndim == 3:
                name = pos.__class__.get_name()
                Diff = coord.representation.DIFFERENTIAL_CLASSES[name]
                vel = Diff(*vel)
            else:
                Diff = rep_nd.NDCartesianDifferential
                vel = Diff(vel)

        # make sure shape is the same
        if pos.shape != vel.shape:
            raise ValueError(
                "Position and velocity must have the same shape "
                f"{pos.shape} vs. {vel.shape}"
            )

        from ..potential.frame import FrameBase

        if frame is not None and not isinstance(frame, FrameBase):
            raise TypeError(
                "Input reference frame must be a FrameBase subclass instance."
            )

        self.pos = pos
        self.vel = vel
        self.frame = frame
        self.ndim = ndim

    def __getitem__(self, slyce):
        return self.__class__(
            pos=self.pos[slyce], vel=self.vel[slyce], frame=self.frame
        )

    def get_components(self, which):
        """
        Get the component name dictionary for the desired object.

        The returned dictionary maps component names on this class to component
        names on the desired object.

        Parameters
        ----------
        which : str
            Can either be ``'pos'`` or ``'vel'`` to get the components for the
            position or velocity object.
        """
        mappings = self.representation_mappings.get(getattr(self, which).__class__, [])

        old_to_new = {}
        for name in getattr(self, which).components:
            for m in mappings:
                if isinstance(m, RegexRepresentationMapping):
                    pattr = re.match(m.repr_name, name)
                    old_to_new[name] = m.new_name.format(*pattr.groups())

                elif m.repr_name == name:
                    old_to_new[name] = m.new_name

        mapping = {}
        for name in getattr(self, which).components:
            mapping[old_to_new.get(name, name)] = name

        return mapping

    @property
    def pos_components(self):
        return self.get_components("pos")

    @property
    def vel_components(self):
        return self.get_components("vel")

    def _get_extra_mappings(self, which):
        mappings = self.representation_mappings.get(getattr(self, which).__class__, [])

        extra = {}
        for m in mappings:
            if m.new_name not in self.get_components(which) and not isinstance(
                m, RegexRepresentationMapping
            ):
                extra[m.new_name] = m.repr_name
        return extra

    def __dir__(self):
        """
        Override the builtin `dir` behavior to include representation and
        differential names.
        """
        dir_values = set(self.pos_components.keys())
        dir_values |= set(self.vel_components.keys())
        dir_values |= set(self._get_extra_mappings("pos").keys())
        dir_values |= set(self._get_extra_mappings("vel").keys())
        dir_values |= set(r.REPRESENTATION_CLASSES.keys())
        dir_values |= set(super().__dir__())
        return sorted(dir_values)

    def __getattr__(self, attr):
        """
        Allow access to attributes on the ``pos`` and ``vel`` representation and
        differential objects.
        """

        # Prevent infinite recursion here.
        if attr.startswith("_"):
            return self.__getattribute__(attr)  # Raise AttributeError.

        # TODO: with >3.5 support, can do:
        # pos_comps = {**self.pos_components,
        #              **self._get_extra_mappings('pos')}
        pos_comps = self.pos_components.copy()
        pos_comps.update(self._get_extra_mappings("pos"))
        if attr in pos_comps:
            return getattr(self.pos, pos_comps[attr])

        # TODO: with >3.5 support, can do:
        # pos_comps = {**self.vel_components,
        #              **self._get_extra_mappings('vel')}
        vel_comps = self.vel_components.copy()
        vel_comps.update(self._get_extra_mappings("vel"))
        if attr in vel_comps:
            return getattr(self.vel, vel_comps[attr])

        if attr in r.REPRESENTATION_CLASSES:
            return self.represent_as(attr)

        return self.__getattribute__(attr)  # Raise AttributeError.

    @property
    def data(self):
        return self.pos.with_differentials(self.vel)

    # ------------------------------------------------------------------------
    # Convert from Cartesian to other representations
    #
    def represent_as(self, new_pos, new_vel=None):
        """
        Represent the position and velocity of the phase-space position in an
        alternate coordinate system. Supports any of the Astropy coordinates
        representation classes.

        Parameters
        ----------
        new_pos : :class:`~astropy.coordinates.BaseRepresentation` or str
            The type of representation to generate. Must be a class (not an
            instance), or the string name of the representation class.
        new_vel : :class:`~astropy.coordinates.BaseDifferential` or str, optional
            Class in which any velocities should be represented. Must be a class
            (not an instance), or the string name of the differential class. If
            None, uses the default differential for the new position class.

        Returns
        -------
        new_psp : :class:`~gala.dynamics.PhaseSpacePosition`
            A new PhaseSpacePosition object with the specified representation.
        """

        if self.ndim != 3:
            raise ValueError("Can only change representation for ndim=3 instances.")

        # get the name of the desired representation
        pos_name = new_pos if isinstance(new_pos, str) else new_pos.get_name()

        if isinstance(new_vel, str):
            vel_name = new_vel
        elif new_vel is None:
            vel_name = pos_name
        else:
            vel_name = new_vel.get_name()

        Representation = coord.representation.REPRESENTATION_CLASSES[pos_name]
        Differential = coord.representation.DIFFERENTIAL_CLASSES[vel_name]

        new_pos = self.pos.represent_as(Representation)
        new_vel = self.vel.represent_as(Differential, self.pos)

        return self.__class__(pos=new_pos, vel=new_vel, frame=self.frame)

    def to_frame(self, frame, current_frame=None, **kwargs):
        """
        Transform to a new reference frame.

        Parameters
        ----------
        frame : `~gala.potential.FrameBase`
            The frame to transform to.
        current_frame : `gala.potential.CFrameBase`
            The current frame the phase-space position is in.
        **kwargs
            Any additional arguments are passed through to the individual frame
            transformation functions (see:
            `~gala.potential.frame.builtin.transformations`).

        Returns
        -------
        psp : `gala.dynamics.PhaseSpacePosition`
            The phase-space position in the new reference frame.

        """

        from ..potential.frame.builtin import transformations as frame_trans

        if self.frame is None and current_frame is None:
            raise ValueError(
                f"If no frame was specified when this {self} was "
                "initialized, you must pass the current frame in "
                "via the current_frame argument to transform to a "
                "new frame."
            )

        if self.frame is not None and current_frame is None:
            current_frame = self.frame

        name1 = current_frame.__class__.__name__.rstrip("Frame").lower()
        name2 = frame.__class__.__name__.rstrip("Frame").lower()
        func_name = f"{name1}_to_{name2}"

        if not hasattr(frame_trans, func_name):
            msg = f"Unsupported frame transformation: {current_frame} to {frame}"
            raise ValueError(msg)
        trans_func = getattr(frame_trans, func_name)

        pos, vel = trans_func(current_frame, frame, self, **kwargs)
        return PhaseSpacePosition(pos=pos, vel=vel, frame=frame)

    def to_coord_frame(self, frame, galactocentric_frame=None, **kwargs):
        """
        Transform the orbit from Galactocentric, cartesian coordinates to
        Heliocentric coordinates in the specified Astropy coordinate frame.

        Parameters
        ----------
        frame : :class:`~astropy.coordinates.BaseCoordinateFrame`
            The frame instance specifying the desired output frame.
            For example, :class:`~astropy.coordinates.ICRS`.
        galactocentric_frame : :class:`~astropy.coordinates.Galactocentric`
            This is the assumed frame that the position and velocity of this
            object are in. The ``Galactocentric`` instand should have parameters
            specifying the position and motion of the sun in the Galactocentric
            frame, but no data.

        Returns
        -------
        c : :class:`~astropy.coordinates.BaseCoordinateFrame`
            An instantiated coordinate frame containing the positions and
            velocities from this object transformed to the specified coordinate
            frame.

        """

        if self.ndim != 3:
            raise ValueError("Can only change representation for ndim=3 instances.")

        if galactocentric_frame is None:
            galactocentric_frame = coord.Galactocentric()

        pos_keys = list(self.pos_components.keys())
        vel_keys = list(self.vel_components.keys())
        if u.one in {getattr(self, pos_keys[0]).unit, getattr(self, vel_keys[0]).unit}:
            raise u.UnitConversionError(
                "Position and velocity must have "
                "dimensioned units to convert to a "
                "coordinate frame."
            )

        # first we need to turn the position into a Galactocentric instance
        gc_c = galactocentric_frame.realize_frame(self.pos.with_differentials(self.vel))
        return gc_c.transform_to(frame)

    # Pseudo-backwards compatibility
    def w(self, units=None):
        """
        Return the full phase-space position as a single array.

        This returns a single array containing the phase-space positions,
        with positions in the first half and velocities in the second half.

        Parameters
        ----------
        units : :class:`~gala.units.UnitSystem`, optional
            The unit system to represent the position and velocity in
            before combining into the full array. If not provided, the
            positions and velocities must be dimensionless.

        Returns
        -------
        w : :class:`~numpy.ndarray`
            A numpy array of all positions and velocities, without units.
            Will have shape ``(2*ndim, ...)`` where the first ``ndim`` rows
            contain positions and the last ``ndim`` rows contain velocities.

        Raises
        ------
        ValueError
            If no units are specified and the position/velocity have units.
        """
        cart = self.cartesian if self.ndim == 3 else self

        xyz = cart.xyz
        d_xyz = cart.v_xyz

        x_unit = xyz.unit
        v_unit = d_xyz.unit
        if (units is None or isinstance(units, DimensionlessUnitSystem)) and (
            x_unit == u.one and v_unit == u.one
        ):
            units = DimensionlessUnitSystem()

        elif units is None:
            raise ValueError("A UnitSystem must be provided.")

        x = xyz.decompose(units).value
        if x.ndim < 2:
            x = atleast_2d(x, insert_axis=1)

        v = d_xyz.decompose(units).value
        if v.ndim < 2:
            v = atleast_2d(v, insert_axis=1)

        return np.vstack((x, v))

    @classmethod
    def from_w(cls, w, units=None, **kwargs):
        """Create a PhaseSpacePosition from a single array of positions and velocities.

        Parameters
        ----------
        w : array_like
            The array of phase-space positions. Should have shape ``(2*ndim, ...)``
            where the first ``ndim`` rows contain positions and the last ``ndim``
            rows contain velocities.
        units : :class:`~gala.units.UnitSystem`, optional
            The unit system that the input position+velocity array, ``w``,
            is represented in. If not provided, the array is assumed to be
            dimensionless.
        **kwargs
            Additional keyword arguments passed to the class initializer.

        Returns
        -------
        obj : :class:`~gala.dynamics.PhaseSpacePosition`
            A new PhaseSpacePosition instance created from the input array.

        """

        w = np.array(w)

        ndim = w.shape[0] // 2
        pos = w[:ndim]
        vel = w[ndim:]

        # TODO: this is bad form - UnitSystem should know what to do with a
        # Dimensionless
        if units is not None and not isinstance(units, DimensionlessUnitSystem):
            units = UnitSystem(units)
            pos *= units["length"]
            vel = vel * units["length"] / units["time"]  # from _core_units

        return cls(pos=pos, vel=vel, **kwargs)

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

        if isinstance(f, str):
            import h5py

            f = h5py.File(f, mode="r")

        if self.frame is not None:
            frame_group = f.create_group("frame")
            frame_group.attrs["module"] = self.frame.__module__
            frame_group.attrs["class"] = self.frame.__class__.__name__

            units = [str(x).encode("utf8") for x in self.frame.units.to_dict().values()]
            frame_group.create_dataset("units", data=units)

            d = frame_group.create_group("parameters")
            for k, par in self.frame.parameters.items():
                quantity_to_hdf5(d, k, par)

        cart = self.represent_as("cartesian")
        quantity_to_hdf5(f, "pos", cart.xyz)
        quantity_to_hdf5(f, "vel", cart.v_xyz)

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
        if isinstance(f, str):
            import h5py

            f = h5py.File(f, mode="r")

        pos = quantity_from_hdf5(f["pos"])
        vel = quantity_from_hdf5(f["vel"])

        frame = None
        if "frame" in f:
            g = f["frame"]

            frame_mod = g.attrs["module"]
            frame_cls = g.attrs["class"]
            frame_units = [u.Unit(x.decode("utf-8")) for x in g["units"]]

            if u.dimensionless_unscaled in frame_units:
                units = DimensionlessUnitSystem()
            else:
                units = UnitSystem(*frame_units)

            pars = {}
            for k in g["parameters"]:
                pars[k] = quantity_from_hdf5(g["parameters/" + k])

            frame_cls = getattr(importlib.import_module(frame_mod), frame_cls)
            frame = frame_cls(units=units, **pars)

        return cls(pos=pos, vel=vel, frame=frame)

    # ------------------------------------------------------------------------
    # Computed dynamical quantities
    #
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
        return 0.5 * self.vel.norm() ** 2

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
        # TODO: check that potential ndim is consistent with here
        return potential.energy(self)

    def energy(self, hamiltonian):
        r"""
        The total energy *per unit mass* (e.g., kinetic + potential):

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
        from gala.potential import Hamiltonian

        hamiltonian = Hamiltonian(hamiltonian)
        return hamiltonian(self)

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
            >>> w = PhaseSpacePosition(pos, vel)
            >>> w.angular_momentum() # doctest: +FLOAT_CMP
            <Quantity [0.        ,0.        ,6.28318531] AU2 / yr>
        """
        cart = self.represent_as(coord.CartesianRepresentation)
        return cart.pos.cross(cart.vel).xyz

    def guiding_radius(self, potential, t=0.0, **root_kwargs):
        """
        Compute the guiding-center radius

        Parameters
        ----------
        potential : `gala.potential.PotentialBase` subclass instance
            The potential to compute the guiding radius in.
        t : quantity-like (optional)
            Time.
        **root_kwargs
            Any additional keyword arguments are passed to `~scipy.optimize.root`.

        Returns
        -------
        Rg : :class:`~astropy.units.Quantity`
            Guiding-center radius.
        """

        R0s = np.atleast_1d(
            np.sqrt(self.x**2 + self.y**2).decompose(potential.units).value
        )
        Lzs = np.atleast_1d(self.angular_momentum()[2].decompose(potential.units).value)
        Rgs = _guiding_radius_helper(R0s, Lzs, potential, t, **root_kwargs)

        return Rgs.reshape(self.shape) * potential.units["length"]

    # ------------------------------------------------------------------------
    # Misc. useful methods
    #
    def _plot_prepare(self, components, units):
        """
        Prepare the ``PhaseSpacePosition`` or subclass for passing to a plotting
        routine to plot all projections of the object.
        """

        # components to plot
        if components is None:
            components = self.pos.components
        n_comps = len(components)

        # if units not specified, get units from the components
        if units is not None:
            if isinstance(units, u.UnitBase):
                units = [units] * n_comps  # global unit

            elif isinstance(units, UnitSystem):
                list_units = []
                for name in components:
                    val = getattr(self, name)
                    list_units.append(units[val.unit.physical_type])
                units = list_units

            elif len(units) != n_comps:
                raise ValueError(
                    "You must specify a unit for each axis, or a "
                    "single unit for all axes."
                )

        labels = []
        x = []
        for i, name in enumerate(components):
            val = getattr(self, name)

            if units is not None:
                val = val.to(units[i])
                unit = units[i]
            else:
                unit = val.unit

            if val.unit != u.one:
                uu = unit.to_string(format="latex_inline")
                unit_str = f" [{uu}]"
            else:
                unit_str = ""

            # Figure out how to fancy display the component name
            if name.startswith("d_"):
                dot = True
                name = name[2:]
            else:
                dot = False

            if name in _greek_letters:
                name = rf"\{name}"

            if dot:
                name = rf"\dot{{{name}}}"

            labels.append(f"${name}$" + unit_str)
            x.append(val.value)

        return x, labels

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
            ``['d_x', 'd_y', 'd_z']``.
        units : `~astropy.units.UnitBase`, iterable, `gala.units.UnitSystem` (optional)
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
            raise ImportError("matplotlib is required for visualization.")
        import matplotlib.pyplot as plt

        if components is None:
            components = self.pos.components

        x, labels = self._plot_prepare(components=components, units=units)

        kwargs.setdefault("plot_function", plt.scatter)
        if kwargs["plot_function"] in {plt.plot, plt.scatter}:
            kwargs.setdefault("marker", ".")
            kwargs.setdefault("labels", labels)
            kwargs.setdefault("plot_function", plt.scatter)
            kwargs.setdefault("autolim", False)

        fig = plot_projections(x, **kwargs)

        if (
            self.pos.get_name() == "cartesian"
            and all(not c.startswith("d_") for c in components)
            and auto_aspect
        ):
            for ax in fig.axes:
                ax.set(aspect="equal", adjustable="datalim")

        return fig

    # ------------------------------------------------------------------------
    # Display
    #
    def __repr__(self):
        return f"<{self.__class__.__name__} {self.pos.get_name()}, dim={self.ndim}, shape={self.pos.shape}>"

    def __str__(self):
        return f"pos={self.pos}\nvel={self.vel}"

    # ------------------------------------------------------------------------
    # Shape and size
    #

    @property
    def shape(self):
        """
        This is *not* the shape of the position or velocity arrays. That is
        accessed by doing, e.g., ``obj.x.shape``.
        """
        return self.pos.shape

    def reshape(self, new_shape):
        """
        Reshape the underlying position and velocity arrays.
        """
        return self.__class__(
            pos=self.pos.reshape(new_shape),
            vel=self.vel.reshape(new_shape),
            frame=self.frame,
        )


def _guiding_radius_rootfunc(R, Lz, potential, t):
    dPhi_dR = potential.c_instance.d_dr(
        np.array([[R[0], 0.0, 0.0]]), potential.G, t=np.array([t])
    )
    vc = np.sqrt(R * np.abs(dPhi_dR))
    return Lz - R * vc


def _guiding_radius_helper(R0s, Lzs, potential, t, **root_kwargs):
    from scipy.optimize import root

    root_kwargs.setdefault("options", {"xtol": 1e-5})
    root_kwargs.setdefault("method", "hybr")

    Rgs = np.zeros_like(R0s)
    for i, (R0, Lz) in enumerate(zip(R0s, Lzs)):
        res = root(
            _guiding_radius_rootfunc, R0, args=(np.abs(Lz), potential, t), **root_kwargs
        )
        if res.success:
            Rgs[i] = res.x[0]
        else:
            Rgs[i] = np.nan

    return Rgs
