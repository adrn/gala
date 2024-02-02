# Standard library
import abc
import copy as pycopy
import uuid
import warnings
from collections import OrderedDict

import astropy.units as u

# Third-party
import numpy as np
from astropy.constants import G
from astropy.utils import isiterable
from astropy.utils.decorators import deprecated

try:
    from scipy.spatial.transform import Rotation
except ImportError as exc:
    raise ImportError(
        "Gala requires scipy>=1.2: make sure you have updated your version of "
        "scipy and try importing gala again."
    ) from exc

# Project
from gala.util import GalaDeprecationWarning

from ...units import DimensionlessUnitSystem
from ...util import ImmutableDict, atleast_2d
from ..common import CommonBase

__all__ = ["PotentialBase", "CompositePotential"]


class PotentialBase(CommonBase, metaclass=abc.ABCMeta):
    """
    A baseclass for defining pure-Python gravitational potentials.

    Subclasses must define (at minimum) a method that evaluates the potential
    energy at a given position ``q`` and time ``t``: ``_energy(q, t)``. For
    integration, the subclasses must also define a method to evaluate the
    gradient, ``_gradient(q, t)``. Optionally, they may also define methods to
    compute the density and hessian: ``_density()``, ``_hessian()``.
    """

    ndim = 3

    def __init__(self, *args, units=None, origin=None, R=None, **kwargs):
        if self._GSL_only:
            from gala._cconfig import GSL_ENABLED

            if not GSL_ENABLED:
                raise ValueError(
                    "Gala was compiled without GSL and so this potential -- "
                    f"{str(self.__class__)} -- will not work.  See the gala "
                    "documentation for more information about installing and "
                    "using GSL with gala: "
                    "http://gala.adrian.pw/en/latest/install.html"
                )

        parameter_values = self._parse_parameter_values(*args, **kwargs)
        self._setup_potential(
            parameters=parameter_values, origin=origin, R=R, units=units
        )

    def _setup_potential(self, parameters, origin=None, R=None, units=None):
        self._units = self._validate_units(units)
        self.parameters = self._prepare_parameters(parameters, self.units)

        try:
            self.G = G.decompose(self.units).value
        except u.UnitConversionError:
            # TODO: this is a convention that and could lead to confusion!
            self.G = 1.0

        if origin is None:
            origin = np.zeros(self.ndim)
        self.origin = self._remove_units(origin)

        if R is not None and self.ndim not in [2, 3]:
            raise NotImplementedError(
                "Gala potentials currently only support "
                "rotations when ndim=2 or ndim=3."
            )

        if R is not None:
            if isinstance(R, Rotation):
                R = R.as_matrix()
            R = np.array(R)

            if R.shape != (self.ndim, self.ndim):
                raise ValueError(
                    "Rotation matrix passed to potential {0} has "
                    "an invalid shape: expected {1}, got {2}".format(
                        self.__class__.__name__, (self.ndim, self.ndim), R.shape
                    )
                )
        self.R = R

    def replicate(self, **kwargs):
        """
        Return a copy of the potential instance with some parameter values
        changed. This always produces copies of any parameter arrays.

        Parameters
        ----------
        **kwargs
            All other keyword arguments are used to overwrite parameter values
            when making the copy.

        Returns
        -------
        replicant : `~gala.potential.PotentialBase` subclass instance
            The replicated potential.
        """
        for k, v in self.parameters.items():
            kwargs.setdefault(k, pycopy.copy(v))

        for k in ["units", "origin", "R"]:
            v = getattr(self, k)
            kwargs.setdefault(k, pycopy.copy(v))

        return self.__class__(**kwargs)

    @classmethod
    def to_sympy(cls):
        """Return a representation of this potential class as a sympy expression

        Returns
        -------
        expr : sympy expression
        vars : dict
            A dictionary of sympy symbols used in the expression.
        """
        raise NotImplementedError(
            "to_sympy() is not implemented for this " f"class {cls}"
        )

    @classmethod
    def to_latex(cls):
        """Return a string LaTeX representation of this potential

        Returns
        -------
        latex_str : str
            The latex expression as a Python string.
        """
        try:
            expr, *_ = cls.to_sympy()
        except NotImplementedError:
            raise NotImplementedError(
                ".to_latex() requires having a .to_sympy() method implemented "
                "on the requesting potential class"
            )

        # testing for this import happens in the sympy method
        import sympy as sy

        return sy.latex(expr)

    ###########################################################################
    # Abstract methods that must be implemented by subclasses
    #
    @abc.abstractmethod
    def _energy(self, q, t=0.0):
        pass

    @abc.abstractmethod
    def _gradient(self, q, t=0.0):
        pass

    def _density(self, q, t=0.0):
        raise NotImplementedError(
            "This Potential has no implemented density " "function."
        )

    def _hessian(self, q, t=0.0):
        raise NotImplementedError("This Potential has no implemented Hessian.")

    ###########################################################################
    # Utility methods
    #
    def _remove_units(self, x):
        """
        Always returns an array. If a Quantity is passed in, it converts to the
        units associated with this object and returns the value.
        """
        if hasattr(x, "unit"):
            x = x.decompose(self.units).value

        else:
            x = np.array(x)

        return x

    def _remove_units_prepare_shape(self, x):
        """
        This is similar to that implemented by
        `gala.potential.common.CommonBase`, but returns just the position if the
        input is a `PhaseSpacePosition`.
        """
        from gala.dynamics import PhaseSpacePosition

        if hasattr(x, "unit"):
            x = x.decompose(self.units).value

        elif isinstance(x, PhaseSpacePosition):
            x = x.cartesian.xyz.decompose(self.units).value

        x = atleast_2d(x, insert_axis=1).astype(np.float64)

        if x.shape[0] != self.ndim:
            raise ValueError(
                f"Input position has ndim={x.shape[0]}, but this potential "
                f"expects an {self.ndim}-dimensional position."
            )

        return x

    ###########################################################################
    # Core methods that use the above implemented functions
    #
    def energy(self, q, t=0.0):
        """
        Compute the potential energy at the given position(s).

        Parameters
        ----------
        q : `~gala.dynamics.PhaseSpacePosition`, `~astropy.units.Quantity`, array_like
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.

        Returns
        -------
        E : `~astropy.units.Quantity`
            The potential energy per unit mass or value of the potential.
        """
        q = self._remove_units_prepare_shape(q)
        orig_shape, q = self._get_c_valid_arr(q)
        t = self._validate_prepare_time(t, q)
        ret_unit = self.units["energy"] / self.units["mass"]

        return self._energy(q, t=t).T.reshape(orig_shape[1:]) * ret_unit

    def gradient(self, q, t=0.0):
        """
        Compute the gradient of the potential at the given position(s).

        Parameters
        ----------
        q : `~gala.dynamics.PhaseSpacePosition`, `~astropy.units.Quantity`, array_like
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.

        Returns
        -------
        grad : `~astropy.units.Quantity`
            The gradient of the potential. Will have the same shape as
            the input position.
        """
        q = self._remove_units_prepare_shape(q)
        orig_shape, q = self._get_c_valid_arr(q)
        t = self._validate_prepare_time(t, q)
        ret_unit = self.units["length"] / self.units["time"] ** 2
        uu = self.units["acceleration"]
        return (self._gradient(q, t=t).T.reshape(orig_shape) * ret_unit).to(uu)

    def density(self, q, t=0.0):
        """
        Compute the density value at the given position(s).

        Parameters
        ----------
        q : `~gala.dynamics.PhaseSpacePosition`, `~astropy.units.Quantity`, array_like
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.

        Returns
        -------
        dens : `~astropy.units.Quantity`
            The potential energy or value of the potential. If the input
            position has shape ``q.shape``, the output energy will have
            shape ``q.shape[1:]``.
        """
        q = self._remove_units_prepare_shape(q)
        orig_shape, q = self._get_c_valid_arr(q)
        t = self._validate_prepare_time(t, q)
        ret_unit = self.units["mass"] / self.units["length"] ** 3
        return (self._density(q, t=t).T * ret_unit).to(self.units["mass density"])

    def hessian(self, q, t=0.0):
        """
        Compute the Hessian of the potential at the given position(s).

        Parameters
        ----------
        q : `~gala.dynamics.PhaseSpacePosition`, `~astropy.units.Quantity`, array_like
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.

        Returns
        -------
        hess : `~astropy.units.Quantity`
            The Hessian matrix of second derivatives of the potential. If the
            input position has shape ``q.shape``, the output energy will have
            shape ``(q.shape[0],q.shape[0]) + q.shape[1:]``. That is, an
            ``n_dim`` by ``n_dim`` array (matrix) for each position.
        """
        if self.R is not None and not np.allclose(
            np.diag(self.R), 1.0, atol=1e-15, rtol=0
        ):
            raise NotImplementedError(
                "Computing Hessian matrices for rotated "
                "potentials is currently not supported."
            )
        q = self._remove_units_prepare_shape(q)
        orig_shape, q = self._get_c_valid_arr(q)
        t = self._validate_prepare_time(t, q)
        ret_unit = 1 / self.units["time"] ** 2
        hess = np.moveaxis(self._hessian(q, t=t), 0, -1)
        return hess.reshape((orig_shape[0], orig_shape[0]) + orig_shape[1:]) * ret_unit

    ###########################################################################
    # Convenience methods that make use the base methods
    #
    def acceleration(self, q, t=0.0):
        """
        Compute the acceleration due to the potential at the given position(s).

        Parameters
        ----------
        q : `~gala.dynamics.PhaseSpacePosition`, `~astropy.units.Quantity`, array_like
            Position to compute the acceleration at.

        Returns
        -------
        acc : `~astropy.units.Quantity`
            The acceleration. Will have the same shape as the input
            position array, ``q``.
        """
        return -self.gradient(q, t=t)

    def mass_enclosed(self, q, t=0.0):
        """
        Estimate the mass enclosed within the given position by assuming the potential
        is spherical.

        Parameters
        ----------
        q : `~gala.dynamics.PhaseSpacePosition`, `~astropy.units.Quantity`, array_like
            Position(s) to estimate the enclossed mass.

        Returns
        -------
        menc : `~astropy.units.Quantity`
            Mass enclosed at the given position(s). If the input position
            has shape ``q.shape``, the output energy will have shape
            ``q.shape[1:]``.
        """
        q = self._remove_units_prepare_shape(q)
        orig_shape, q = self._get_c_valid_arr(q)
        t = self._validate_prepare_time(t, q)

        # small step-size in direction of q
        h = 1e-3  # MAGIC NUMBER

        # Radius
        r = np.sqrt(np.sum(q**2, axis=1))

        epsilon = h * q / r[:, np.newaxis]

        dPhi_dr_plus = self._energy(q + epsilon, t=t)
        dPhi_dr_minus = self._energy(q - epsilon, t=t)
        diff = dPhi_dr_plus - dPhi_dr_minus

        if isinstance(self.units, DimensionlessUnitSystem):
            Gee = 1.0
        else:
            Gee = G.decompose(self.units).value

        Menc = np.abs(r * r * diff / Gee / (2.0 * h))
        Menc = Menc.reshape(orig_shape[1:])

        sgn = 1.0
        if "m" in self.parameters and self.parameters["m"] < 0:
            sgn = -1.0

        return sgn * Menc * self.units["mass"]

    def circular_velocity(self, q, t=0.0):
        """
        Estimate the circular velocity at the given position assuming the
        potential is spherical.

        Parameters
        ----------
        q : array_like, numeric
            Position(s) to estimate the circular velocity.

        Returns
        -------
        vcirc : `~astropy.units.Quantity`
            Circular velocity at the given position(s). If the input position
            has shape ``q.shape``, the output energy will have shape
            ``q.shape[1:]``.

        """
        q = self._remove_units_prepare_shape(q)

        # Radius
        r = np.sqrt(np.sum(q**2, axis=0)) * self.units["length"]
        dPhi_dxyz = self.gradient(q, t=t)
        dPhi_dr = np.sum(dPhi_dxyz * q / r.value, axis=0)

        return self.units.decompose(np.sqrt(r * np.abs(dPhi_dr)))

    ###########################################################################
    # Python special methods
    #
    def __call__(self, q):
        return self.energy(q)

    def __add__(self, other):
        if not isinstance(other, PotentialBase):
            raise TypeError(
                f"Cannot add a {self.__class__.__name__} to a "
                f"{other.__class__.__name__}"
            )

        new_pot = CompositePotential()

        if isinstance(self, CompositePotential):
            for k, v in self.items():
                new_pot[k] = v

        else:
            k = str(uuid.uuid4())
            new_pot[k] = self

        if isinstance(other, CompositePotential):
            for k, v in self.items():
                if k in new_pot:
                    raise KeyError(
                        f'Potential component "{k}" already exists '
                        "-- duplicate key provided in potential "
                        "addition"
                    )
                new_pot[k] = v

        else:
            k = str(uuid.uuid4())
            new_pot[k] = other

        return new_pot

    ###########################################################################
    # Convenience methods that do fancy things
    #
    def plot_contours(
        self,
        grid,
        t=0.0,
        filled=True,
        ax=None,
        labels=None,
        subplots_kw=dict(),
        **kwargs,
    ):
        """
        Plot equipotentials contours. Computes the potential energy on a grid
        (specified by the array `grid`).

        .. warning:: Right now the grid input must be arrays and must already
            be in the unit system of the potential. Quantity support is coming...

        Parameters
        ----------
        grid : tuple
            Coordinate grids or slice value for each dimension. Should be a
            tuple of 1D arrays or numbers.
        t : quantity-like (optional)
            The time to evaluate at.
        filled : bool (optional)
            Use :func:`~matplotlib.pyplot.contourf` instead of
            :func:`~matplotlib.pyplot.contour`. Default is ``True``.
        ax : matplotlib.Axes (optional)
        labels : iterable (optional)
            List of axis labels.
        subplots_kw : dict
            kwargs passed to matplotlib's subplots() function if an axes object
            is not specified.
        kwargs : dict
            kwargs passed to either contourf() or plot().

        Returns
        -------
        fig : `~matplotlib.Figure`

        """

        import matplotlib.pyplot as plt
        from matplotlib import cm

        # figure out which elements are iterable, which are numeric
        _grids = []
        _slices = []
        for ii, g in enumerate(grid):
            if isiterable(g):
                _grids.append((ii, g))
            else:
                _slices.append((ii, g))

        # figure out the dimensionality
        ndim = len(_grids)

        # if ndim > 2, don't know how to handle this!
        if ndim > 2:
            raise ValueError(
                "ndim > 2: you can only make contours on a 2D grid. For other "
                "dimensions, you have to specify values to slice."
            )

        if ax is None:
            # default figsize
            fig, ax = plt.subplots(1, 1, **subplots_kw)
        else:
            fig = ax.figure

        if ndim == 1:
            # 1D curve
            x1 = _grids[0][1]
            r = np.zeros((len(_grids) + len(_slices), len(x1)))
            r[_grids[0][0]] = x1

            for ii, slc in _slices:
                r[ii] = slc

            Z = self.energy(r * self.units["length"], t=t).value
            ax.plot(x1, Z, **kwargs)

            if labels is not None:
                ax.set_xlabel(labels[0])
                ax.set_ylabel("potential")
        else:
            # 2D contours
            x1, x2 = np.meshgrid(_grids[0][1], _grids[1][1])
            shp = x1.shape
            x1, x2 = x1.ravel(), x2.ravel()

            r = np.zeros((len(_grids) + len(_slices), len(x1)))
            r[_grids[0][0]] = x1
            r[_grids[1][0]] = x2

            for ii, slc in _slices:
                r[ii] = slc

            Z = self.energy(r * self.units["length"], t=t).value

            # make default colormap not suck
            cmap = kwargs.pop("cmap", cm.Blues)
            if filled:
                ax.contourf(
                    x1.reshape(shp),
                    x2.reshape(shp),
                    Z.reshape(shp),
                    cmap=cmap,
                    **kwargs,
                )
            else:
                ax.contour(
                    x1.reshape(shp),
                    x2.reshape(shp),
                    Z.reshape(shp),
                    cmap=cmap,
                    **kwargs,
                )

            if labels is not None:
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1])

        return fig

    def plot_density_contours(
        self,
        grid,
        t=0.0,
        filled=True,
        ax=None,
        labels=None,
        subplots_kw=dict(),
        **kwargs,
    ):
        """
        Plot density contours. Computes the density on a grid
        (specified by the array `grid`).

        .. warning::

            For now, the grid input must be arrays and must already be in
            the unit system of the potential. Quantity support is coming...

        Parameters
        ----------
        grid : tuple
            Coordinate grids or slice value for each dimension. Should be a
            tuple of 1D arrays or numbers.
        t : quantity-like (optional)
            The time to evaluate at.
        filled : bool (optional)
            Use :func:`~matplotlib.pyplot.contourf` instead of
            :func:`~matplotlib.pyplot.contour`. Default is ``True``.
        ax : matplotlib.Axes (optional)
        labels : iterable (optional)
            List of axis labels.
        subplots_kw : dict
            kwargs passed to matplotlib's subplots() function if an axes object
            is not specified.
        kwargs : dict
            kwargs passed to either contourf() or plot().

        Returns
        -------
        fig : `~matplotlib.Figure`

        """

        import matplotlib.pyplot as plt
        from matplotlib import cm

        # figure out which elements are iterable, which are numeric
        _grids = []
        _slices = []
        for ii, g in enumerate(grid):
            if isiterable(g):
                _grids.append((ii, g))
            else:
                _slices.append((ii, g))

        # figure out the dimensionality
        ndim = len(_grids)

        # if ndim > 2, don't know how to handle this!
        if ndim > 2:
            raise ValueError(
                "ndim > 2: you can only make contours on a 2D grid. For other "
                "dimensions, you have to specify values to slice."
            )

        if ax is None:
            # default figsize
            fig, ax = plt.subplots(1, 1, **subplots_kw)
        else:
            fig = ax.figure

        if ndim == 1:
            # 1D curve
            x1 = _grids[0][1]
            r = np.zeros((len(_grids) + len(_slices), len(x1)))
            r[_grids[0][0]] = x1

            for ii, slc in _slices:
                r[ii] = slc

            Z = self.density(r * self.units["length"], t=t).value
            ax.plot(x1, Z, **kwargs)

            if labels is not None:
                ax.set_xlabel(labels[0])
                ax.set_ylabel("potential")
        else:
            # 2D contours
            x1, x2 = np.meshgrid(_grids[0][1], _grids[1][1])
            shp = x1.shape
            x1, x2 = x1.ravel(), x2.ravel()

            r = np.zeros((len(_grids) + len(_slices), len(x1)))
            r[_grids[0][0]] = x1
            r[_grids[1][0]] = x2

            for ii, slc in _slices:
                r[ii] = slc

            Z = self.density(r * self.units["length"], t=t).value

            # make default colormap not suck
            cmap = kwargs.pop("cmap", cm.Blues)
            if filled:
                ax.contourf(
                    x1.reshape(shp),
                    x2.reshape(shp),
                    Z.reshape(shp),
                    cmap=cmap,
                    **kwargs,
                )
            else:
                ax.contour(
                    x1.reshape(shp),
                    x2.reshape(shp),
                    Z.reshape(shp),
                    cmap=cmap,
                    **kwargs,
                )

            # cs.cmap.set_under('w')
            # cs.cmap.set_over('k')

            if labels is not None:
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1])

        return fig

    def plot_rotation_curve(self, R_grid, t=0.0, ax=None, labels=None, **plot_kwargs):
        """
        Plot the rotation curve or circular velocity curve for this potential on the
        input grid of cylindrical radii.

        Parameters
        ----------
        R_grid : array-like
            A grid of radius values to compute the rotation curve at. This should be a
            one-dimensional grid.
        t : quantity-like (optional)
            The time to evaluate at.
        ax : matplotlib.Axes (optional)
        labels : iterable (optional)
            List of axis labels. Set to False to disable adding labels.
        plot_kwargs : dict
            kwargs passed to plot().

        Returns
        -------
        fig : `~matplotlib.Figure`
        ax : `~matplotlib.Axes`

        """

        if not hasattr(R_grid, "unit"):
            R_grid = R_grid * self.units["length"]

        xyz = np.zeros((3,) + R_grid.shape) * self.units["length"]
        xyz[0] = R_grid

        vcirc = self.circular_velocity(xyz, t=t)

        if labels is None:
            labels = [
                f"$R$ [{self.units['length']:latex_inline}]",
                r"$v_{\rm circ}$ " + f"[{self.units['speed']:latex_inline}]",
            ]

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if labels is not False:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])

        plot_kwargs.setdefault("marker", "")
        plot_kwargs.setdefault("linestyle", plot_kwargs.pop("ls", "-"))
        plot_kwargs.setdefault("linewidth", plot_kwargs.pop("lw", 1))

        ax.plot(
            R_grid.to_value(self.units["length"]),
            vcirc.to_value(self.units["speed"]),
            **plot_kwargs,
        )

        return fig, ax

    def integrate_orbit(self, *args, **kwargs):
        """
        Integrate an orbit in the current potential using the integrator class
        provided. Uses same time specification as `Integrator()` -- see
        the documentation for `gala.integrate` for more information.

        Parameters
        ----------
        w0 : `~gala.dynamics.PhaseSpacePosition`, array_like
            Initial conditions.
        Integrator : `~gala.integrate.Integrator` (optional)
            Integrator class to use.
        Integrator_kwargs : dict (optional)
            Any extra keyword argumets to pass to the integrator class
            when initializing. Only works in non-Cython mode.
        cython_if_possible : bool (optional)
            If there is a Cython version of the integrator implemented,
            and the potential object has a C instance, using Cython
            will be *much* faster.
        store_all : bool (optional)
            Controls whether to store the phase-space position at all intermediate
            timesteps. Set to False to store only the final values (i.e. the
            phase-space position(s) at the final timestep). Default is True.
        **time_spec
            Specification of how long to integrate. See documentation
            for `~gala.integrate.parse_time_specification`.

        Returns
        -------
        orbit : `~gala.dynamics.Orbit`

        """
        from ..hamiltonian import Hamiltonian

        return Hamiltonian(self).integrate_orbit(*args, **kwargs)

    def total_energy(self, x, v):
        """
        Compute the total energy (per unit mass) of a point in phase-space
        in this potential. Assumes the last axis of the input position /
        velocity is the dimension axis, e.g., for 100 points in 3-space,
        the arrays should have shape (100, 3).

        Parameters
        ----------
        x : array_like, numeric
            Position.
        v : array_like, numeric
            Velocity.
        """
        warnings.warn(
            "Use the energy methods on Orbit objects instead. In a future "
            "release this will be removed.",
            GalaDeprecationWarning,
        )

        v = atleast_2d(v, insert_axis=1)
        return self.energy(x) + 0.5 * np.sum(v**2, axis=0)

    def save(self, f):
        """
        Save the potential to a text file. See :func:`~gala.potential.save`
        for more information.

        Parameters
        ----------
        f : str, file_like
            A filename or file-like object to write the input potential object to.

        """
        from .io import save

        save(self, f)

    @property
    def units(self):
        return self._units

    def replace_units(self, units, copy=True):
        """Change the unit system of this potential.

        Parameters
        ----------
        units : `~gala.units.UnitSystem`
            Set of non-reducable units that specify (at minimum) the
            length, mass, time, and angle units.
        copy : bool (optional)
            If True, returns a copy, if False, changes this object.
        """
        if copy:
            pot = pycopy.deepcopy(self)
        else:
            pot = self

        # TODO: this is repeated code - see equivalent in cpotential.pyx
        tmp = [
            isinstance(units, DimensionlessUnitSystem),
            isinstance(self.units, DimensionlessUnitSystem),
        ]
        if not all(tmp) and any(tmp):
            raise ValueError(
                "Cannot replace a dimensionless unit system with "
                "a unit system with physical units, or vice versa"
            )

        PotentialBase.__init__(
            pot, origin=self.origin, R=self.R, units=units, **self.parameters
        )

        return pot

    ###########################################################################
    # Deprecated methods
    #
    def _value(self, q, t=0.0):
        warnings.warn("Use `_energy()` instead.", GalaDeprecationWarning)
        return self._energy(q, t=t)

    def value(self, *args, **kwargs):
        __doc__ = self.energy.__doc__  # noqa
        warnings.warn("Use `energy()` instead.", GalaDeprecationWarning)
        return self.energy(*args, **kwargs)

    ###########################################################################
    # Interoperability with other packages
    #
    @deprecated(
        since="v1.8",
        message="This has been replaced by a more general interoperability framework.",
        alternative="interop",
    )
    def to_galpy_potential(self, ro=None, vo=None):
        """Convert a Gala potential to a Galpy potential instance

        Parameters
        ----------
        ro : quantity-like (optional)
        vo : quantity-like (optional)
        """
        return self.as_interop("galpy", ro=ro, vo=vo)

    def as_interop(self, package, **kwargs):
        """Interoperability with other Galactic dynamics packages

        Parameters
        ----------
        package : str
            The package to export the potential to. Currently supported packages are
            ``"galpy"`` and ``"agama"``.
        kwargs
            Any additional keyword arguments are passed to the interop function.
        """
        if package == "galpy":
            from .interop import gala_to_galpy_potential

            kwargs.setdefault("ro", None)
            kwargs.setdefault("vo", None)
            return gala_to_galpy_potential(self, **kwargs)
        elif package == "agama":
            import agama

            from .interop import gala_to_agama_potential

            agama_pot = gala_to_agama_potential(self, **kwargs)
            if not isinstance(agama_pot, agama.Potential):
                agama_pot = agama.Potential(*agama_pot)
            return agama_pot
        else:
            raise ValueError(f"Unsupported package: {package}")


class CompositePotential(PotentialBase, OrderedDict):
    """
    A potential composed of several distinct components. For example,
    two point masses or a galactic disk and halo, each with their own
    potential model.

    A `CompositePotential` is created like a Python dictionary, e.g.::

        >>> p1 = SomePotential(func1) # doctest: +SKIP
        >>> p2 = SomePotential(func2) # doctest: +SKIP
        >>> cp = CompositePotential(component1=p1, component2=p2) # doctest: +SKIP

    This object actually acts like a dictionary, so if you want to
    preserve the order of the potential components, use::

        >>> cp = CompositePotential() # doctest: +SKIP
        >>> cp['component1'] = p1 # doctest: +SKIP
        >>> cp['component2'] = p2 # doctest: +SKIP

    You can also use any of the built-in `Potential` classes as
    components::

        >>> from gala.potential import HernquistPotential
        >>> cp = CompositePotential()
        >>> cp['spheroid'] = HernquistPotential(m=1E11, c=10.,
        ...                                     units=(u.kpc, u.Myr, u.Msun, u.radian))

    """

    def __init__(self, *args, **kwargs):
        self._units = None
        self.ndim = None

        if len(args) > 0 and isinstance(args[0], list):
            for k, v in args[0]:
                kwargs[k] = v
        else:
            for i, v in args:
                kwargs[str(i)] = v

        self.lock = False
        for v in kwargs.values():
            self._check_component(v)

        OrderedDict.__init__(self, **kwargs)

        self.R = None  # TODO: this is a little messy

    def __setitem__(self, key, value):
        self._check_component(value)
        super(CompositePotential, self).__setitem__(key, value)

    def _check_component(self, p):
        if not isinstance(p, PotentialBase):
            raise TypeError(
                "Potential components may only be Potential "
                "objects, not {0}.".format(type(p))
            )

        if self.units is None:
            self._units = p.units
            self.ndim = p.ndim

        else:
            if sorted([str(x) for x in self.units]) != sorted(
                [str(x) for x in p.units]
            ):
                raise ValueError(
                    "Unit system of new potential component must "
                    "match unit systems of other potential "
                    "components."
                )

            if p.ndim != self.ndim:
                raise ValueError(
                    "All potential components must have the same "
                    "number of phase-space dimensions ({} in this "
                    "case)".format(self.ndim)
                )

        if self.lock:
            raise ValueError(
                "Potential object is locked - new components can "
                "only be added to unlocked potentials."
            )

    @property
    def parameters(self):
        params = dict()
        for k, v in self.items():
            params[k] = v.parameters
        return ImmutableDict(**params)

    def replace_units(self, units):
        """Change the unit system of this potential.

        Parameters
        ----------
        units : `~gala.units.UnitSystem`
            Set of non-reducable units that specify (at minimum) the
            length, mass, time, and angle units.
        """
        _lock = self.lock
        pots = self.__class__()

        pots._units = None
        pots.lock = False

        for k, v in self.items():
            pots[k] = v.replace_units(units)

        pots.lock = _lock
        return pots

    def _energy(self, q, t=0.0):
        return np.sum([p._energy(q, t) for p in self.values()], axis=0)

    def _gradient(self, q, t=0.0):
        return np.sum([p._gradient(q, t) for p in self.values()], axis=0)

    def _hessian(self, w, t=0.0):
        return np.sum([p._hessian(w, t) for p in self.values()], axis=0)

    def _density(self, q, t=0.0):
        return np.sum([p._density(q, t) for p in self.values()], axis=0)

    def __repr__(self):
        return "<CompositePotential {}>".format(",".join(self.keys()))

    def replicate(self, **kwargs):
        """
        Return a copy of the potential instance with some parameter values
        changed. This always produces copies of any parameter arrays.

        Parameters
        ----------
        **kwargs
            All other keyword arguments are used to overwrite parameter values
            when making the copy. The keywords passed in should be the same as
            the potential component names, so you can pass in dictionaries to set
            parameters for different subcomponents of this composite potential.

        Returns
        -------
        replicant : `~gala.potential.PotentialBase` subclass instance
            The replicated potential.
        """
        obj = pycopy.copy(self)

        # disable potential lock
        lock = obj.lock
        obj.lock = False

        for k, v in kwargs.items():
            obj[k] = self[k].replicate(**v)

        obj.lock = lock
        return obj


_potential_docstring = """units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.
    origin : `~astropy.units.Quantity` (optional)
        The origin of the potential, the default being 0.
    R : `~scipy.spatial.transform.Rotation`, array_like (optional)
        A Scipy ``Rotation`` object or an array representing a rotation matrix
        that specifies a rotation of the potential. This is applied *after* the
        origin shift. Default is the identity matrix.
"""
