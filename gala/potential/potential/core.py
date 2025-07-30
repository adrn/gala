import abc
import copy as pycopy
import uuid
import warnings
from collections import OrderedDict

import astropy.units as u
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


from gala.util import GalaDeprecationWarning

from ...units import DimensionlessUnitSystem
from ...util import ImmutableDict, atleast_2d
from ..common import CommonBase

__all__ = ["CompositePotential", "PotentialBase"]


class PotentialBase(CommonBase, metaclass=abc.ABCMeta):
    """
    A base class for defining gravitational potentials in Gala.

    This abstract base class provides the foundation for all gravitational
    potential models in Gala. It handles unit conversions, coordinate
    transformations, and provides a consistent interface for computing
    gravitational forces, energies, and related quantities.

    Subclasses must implement the abstract methods ``_energy(q, t)`` and
    ``_gradient(q, t)`` that compute the potential energy and its gradient
    (negative acceleration) at position ``q`` and time ``t``. Optionally,
    subclasses may implement ``_density(q, t)`` and ``_hessian(q, t)`` to
    provide mass density and second derivative information.

    Parameters
    ----------
    units : `~gala.units.UnitSystem`, optional
        Set of non-reducible units that specify (at minimum) the
        length, mass, time, and angle units. If not specified, the default
        unit system will be used.
    origin : array_like, optional
        The origin of the potential in Cartesian coordinates. Default is
        the origin ``[0, 0, 0]``.
    R : array_like, `~scipy.spatial.transform.Rotation`, optional
        Rotation matrix or `~scipy.spatial.transform.Rotation` object to
        rotate the reference frame of the potential. If specified, the
        potential will be evaluated in the rotated coordinate system.

    Attributes
    ----------
    ndim : int
        Number of spatial dimensions (default: 3).
    parameters : `~gala.util.ImmutableDict`
        Dictionary of potential parameters with associated units.
    units : `~gala.units.UnitSystem`
        The unit system used by the potential.
    G : float
        Gravitational constant in the potential's unit system.
    origin : array_like
        The origin of the potential coordinate system.
    R : array_like, optional
        Rotation matrix for the potential coordinate system.

    Notes
    -----
    The potential is evaluated in a coordinate system that may be shifted
    (via ``origin``) and/or rotated (via ``R``) relative to the input
    coordinates. The transformation is applied as:
    ``q_transformed = R @ (q - origin)``.
    """

    ndim = 3

    def __init__(self, *args, units=None, origin=None, R=None, **kwargs):
        if self._GSL_only:
            from gala._cconfig import GSL_ENABLED

            if not GSL_ENABLED:
                raise ValueError(
                    "Gala was compiled without GSL and so this potential -- "
                    f"{self.__class__!s} -- will not work.  See the gala "
                    "documentation for more information about installing and "
                    "using GSL with gala: "
                    "http://gala.adrian.pw/en/latest/install.html"
                )

        if self._EXP_only:
            from gala._cconfig import EXP_ENABLED

            if not EXP_ENABLED:
                raise ValueError(
                    "Gala was compiled without EXP and so this potential -- "
                    f"{self.__class__!s} -- will not work.  See the gala "
                    "documentation for more information about installing and "
                    "using EXP with gala: "
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

        if R is not None and self.ndim not in {2, 3}:
            raise NotImplementedError(
                "Gala potentials currently only support "
                "rotations when ndim=2 or ndim=3."
            )

        if R is not None:
            if isinstance(R, Rotation):
                R = R.as_matrix()
            R = np.array(R)

            if R.shape != (self.ndim, self.ndim):
                msg = (
                    f"Rotation matrix passed to potential {self.__class__.__name__} has "
                    f"an invalid shape: expected {(self.ndim, self.ndim)}, got {R.shape}"
                )
                raise ValueError(msg)
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
        raise NotImplementedError(f"to_sympy() is not implemented for this class {cls}")

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
        except NotImplementedError as e:
            raise NotImplementedError(
                ".to_latex() requires having a .to_sympy() method implemented "
                "on the requesting potential class"
            ) from e

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
        raise NotImplementedError("This Potential has no implemented density function.")

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
        return x.decompose(self.units).value if hasattr(x, "unit") else np.array(x)

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

    def _reapply_units_and_shape(self, x, ptype, shape, conv_unit=None):
        """
        This is the inverse of _remove_units_prepare_shape. It takes the output of one
        of the C functions below and reapplies units and the original shape.
        ptype is an Astropy PhysicalType object
        """
        x = np.moveaxis(x, 0, -1)
        if isinstance(ptype, u.PhysicalType):
            uu = self.units[ptype]
        elif isinstance(ptype, str):
            uu = self.units[u.get_physical_type(ptype)]
        elif isinstance(ptype, u.UnitBase):
            uu = ptype
        else:
            raise ValueError(
                f"ptype must be a PhysicalType, str, or UnitBase object. "
                f"Got {ptype} instead."
            )
        x = x.reshape(shape) * uu
        if conv_unit is None:
            return x
        return x.to(conv_unit)

    ###########################################################################
    # Core methods that use the above implemented functions
    #
    def energy(self, q, t=0.0):
        """
        Compute the gravitational potential energy at the given position(s).

        The potential energy per unit mass is evaluated at the specified
        position(s) and time.

        Parameters
        ----------
        q : `~gala.dynamics.PhaseSpacePosition`, `~astropy.units.Quantity`, array_like
            Position(s) at which to evaluate the potential. If the input
            has no units (i.e., is an `~numpy.ndarray`), it is assumed to
            be in the same unit system as the potential. Shape should be
            ``(n_dim,)`` for a single position or ``(n_dim, n_positions)``
            for multiple positions.
        t : numeric, `~astropy.units.Quantity`, optional
            Time at which to evaluate the potential. Default is 0.

        Returns
        -------
        E : `~astropy.units.Quantity`
            The gravitational potential energy per unit mass. For input
            shape ``(n_dim, n_positions)``, returns shape ``(n_positions,)``.
            Units are specific energy (e.g., m²/s² in SI units).

        Notes
        -----
        The potential energy is related to the gravitational acceleration
        by :math:`\\vec{a} = -\\nabla \\phi`, where φ is the potential
        energy per unit mass.
        """
        q = self._remove_units_prepare_shape(q)
        orig_shape, q = self._get_c_valid_arr(q)
        t = self._validate_prepare_time(t, q)
        return self._reapply_units_and_shape(
            self._energy(q, t=t),
            ptype=u.get_physical_type("energy") / u.get_physical_type("mass"),
            shape=orig_shape[1:],
        )

    def gradient(self, q, t=0.0):
        """
        Compute the gradient of the gravitational potential.

        Parameters
        ----------
        q : `~gala.dynamics.PhaseSpacePosition`, `~astropy.units.Quantity`, array_like
            Position(s) at which to evaluate the potential gradient. If the
            input has no units (i.e., is an `~numpy.ndarray`), it is assumed
            to be in the same unit system as the potential. Shape should be
            ``(n_dim,)`` for a single position or ``(n_dim, n_positions)``
            for multiple positions.
        t : numeric, `~astropy.units.Quantity`, optional
            Time at which to evaluate the potential gradient. Default is 0.

        Returns
        -------
        grad : `~astropy.units.Quantity`
            The gradient of the gravitational potential. Has the same shape
            as the input position array ``q``. Units are acceleration
            (e.g., m/s² in SI units). To get gravitational acceleration,
            use ``acceleration()`` or compute ``-gradient()``.

        See Also
        --------
        acceleration : Compute gravitational acceleration (negative gradient).

        Notes
        -----
        The relationship between potential φ, gradient, and acceleration is:

        .. math::
            \\vec{a} = -\\nabla \\phi = -\\frac{\\partial \\phi}{\\partial \\vec{q}}
        """
        q = self._remove_units_prepare_shape(q)
        orig_shape, q = self._get_c_valid_arr(q)
        t = self._validate_prepare_time(t, q)
        return self._reapply_units_and_shape(
            self._gradient(q, t=t), u.get_physical_type("acceleration"), orig_shape
        )

    def density(self, q, t=0.0):
        """
        Compute the mass density at the given position(s).

        For potentials that have an associated mass distribution, this method
        computes the mass density rho(q, t) at the specified positions and time.
        The density is related to the potential via Poisson's equation:
        :math:`\\nabla^2 \\phi = 4\\pi G \\rho`.

        Parameters
        ----------
        q : `~gala.dynamics.PhaseSpacePosition`, `~astropy.units.Quantity`, array_like
            Position(s) at which to evaluate the mass density. If the input
            has no units (i.e., is an `~numpy.ndarray`), it is assumed to
            be in the same unit system as the potential. Shape should be
            ``(n_dim,)`` for a single position or ``(n_dim, n_positions)``
            for multiple positions.
        t : numeric, `~astropy.units.Quantity`, optional
            Time at which to evaluate the mass density. Default is 0.

        Returns
        -------
        dens : `~astropy.units.Quantity`
            The mass density at the specified position(s). For input
            shape ``(n_dim, n_positions)``, returns shape ``(n_positions,)``.
            Units are mass density (e.g., kg/m³ in SI units).

        Notes
        -----
        Not all potential models have an implemented density function.
        For potentials without a density implementation, this method
        will raise a ``NotImplementedError``.

        The density is computed using the relationship with the potential's
        Laplacian (when available) or from the underlying mass model.

        Raises
        ------
        NotImplementedError
            If the potential does not have an implemented density function.
        """
        q = self._remove_units_prepare_shape(q)
        orig_shape, q = self._get_c_valid_arr(q)
        t = self._validate_prepare_time(t, q)
        return self._reapply_units_and_shape(
            self._density(q, t=t), u.get_physical_type("mass density"), orig_shape[1:]
        )

    def hessian(self, q, t=0.0):
        """
        Compute the Hessian matrix of the gravitational potential.

        The Hessian matrix contains the second partial derivatives of the
        potential: :math:`H_{ij} = \\frac{\\partial^2 \\phi}{\\partial q_i \\partial q_j}`.
        This is useful for stability analysis, computing tidal tensors, and
        orbital frequency analysis.

        Parameters
        ----------
        q : `~gala.dynamics.PhaseSpacePosition`, `~astropy.units.Quantity`, array_like
            Position(s) at which to evaluate the Hessian matrix. If the input
            has no units (i.e., is an `~numpy.ndarray`), it is assumed to
            be in the same unit system as the potential. Shape should be
            ``(n_dim,)`` for a single position or ``(n_dim, n_positions)``
            for multiple positions.
        t : numeric, `~astropy.units.Quantity`, optional
            Time at which to evaluate the Hessian matrix. Default is 0.

        Returns
        -------
        hess : `~astropy.units.Quantity`
            The Hessian matrix of second derivatives. For input shape
            ``(n_dim, n_positions)``, returns shape
            ``(n_dim, n_dim, n_positions)``. Each ``n_dim x n_dim`` slice
            corresponds to the Hessian matrix at one position. Units are
            acceleration per length (e.g., s⁻² in SI units).

        Notes
        -----
        Computing Hessian matrices for rotated potentials (when ``R`` is
        not the identity matrix) is currently not supported and will raise
        a ``NotImplementedError``.

        Not all potential models have an implemented Hessian function.
        For potentials without a Hessian implementation, this method
        will raise a ``NotImplementedError``.

        The Hessian matrix is symmetric for time-independent potentials.

        Raises
        ------
        NotImplementedError
            If the potential does not have an implemented Hessian function,
            or if the potential is rotated (``R`` is not the identity).
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
        return self._reapply_units_and_shape(
            self._hessian(q, t=t),
            u.get_physical_type("frequency drift"),
            (orig_shape[0], orig_shape[0], *orig_shape[1:]),
        )

    ###########################################################################
    # Convenience methods that make use the base methods
    #
    def acceleration(self, q, t=0.0):
        """
        Compute the gravitational acceleration at the given position(s).

        The gravitational acceleration is the negative gradient of the
        potential: :math:`\\vec{a} = -\\nabla \\phi`. This is the
        acceleration experienced by a test particle in the gravitational field.

        Parameters
        ----------
        q : `~gala.dynamics.PhaseSpacePosition`, `~astropy.units.Quantity`, array_like
            Position(s) at which to compute the gravitational acceleration.
            If the input has no units (i.e., is an `~numpy.ndarray`), it is
            assumed to be in the same unit system as the potential.
        t : numeric, `~astropy.units.Quantity`, optional
            Time at which to evaluate the acceleration. Default is 0.

        Returns
        -------
        acc : `~astropy.units.Quantity`
            The gravitational acceleration vector(s). Has the same shape as
            the input position array ``q``. Units are acceleration
            (e.g., m/s² in SI units).

        See Also
        --------
        gradient : Compute the potential gradient (negative acceleration).

        Notes
        -----
        This method is equivalent to ``-self.gradient(q, t)`` and is provided
        for convenience in orbital integration and dynamics calculations.
        """
        return -self.gradient(q, t=t)

    def mass_enclosed(self, q, t=0.0):
        """
        Estimate the mass enclosed within spherical radius at given position(s).

        This method estimates the enclosed mass by assuming spherical symmetry
        and using the relation :math:`M_{\\rm enc}(r) = r^2 |dΦ/dr| / G`, where
        the radial derivative is computed numerically using finite differences.

        Parameters
        ----------
        q : `~gala.dynamics.PhaseSpacePosition`, `~astropy.units.Quantity`, array_like
            Position(s) at which to estimate the enclosed mass. The enclosed
            mass is computed at the spherical radius corresponding to each
            position. If the input has no units, it is assumed to be in the
            same unit system as the potential.
        t : numeric, `~astropy.units.Quantity`, optional
            Time at which to evaluate the enclosed mass. Default is 0.

        Returns
        -------
        menc : `~astropy.units.Quantity`
            Mass enclosed within the spherical radius at each position.
            For input shape ``(n_dim, n_positions)``, returns shape
            ``(n_positions,)``. Units are mass (e.g., kg in SI units).

        Notes
        -----
        This method assumes the potential is approximately spherically
        symmetric. The enclosed mass is estimated using a finite difference
        approximation to the radial derivative of the potential.

        For potentials with negative mass parameters (e.g., some composite
        models), the sign is handled appropriately.

        The calculation uses the relation derived from Gauss's law:

        .. math::
            M_{\\rm enc}(r) = \\frac{r^2}{G} \\left| \\frac{d\\Phi}{dr} \\right|
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

        sgn = 1.0
        if "m" in self.parameters and self.parameters["m"] < 0:
            sgn = -1.0

        return self._reapply_units_and_shape(
            sgn * Menc, u.get_physical_type("mass"), orig_shape[1:]
        )

    def circular_velocity(self, q, t=0.0):
        """
        Estimate the circular velocity at given position(s) assuming spherical symmetry.

        The circular velocity is the speed required for a circular orbit at
        the given radius in a spherically symmetric potential. It is computed
        using :math:`v_{\\rm circ}(r) = \\sqrt{r |dΦ/dr|}`, where the radial
        derivative is evaluated from the potential gradient.

        Parameters
        ----------
        q : `~gala.dynamics.PhaseSpacePosition`, `~astropy.units.Quantity`, array_like
            Position(s) at which to estimate the circular velocity. The
            calculation uses the spherical radius from the origin. If the
            input has no units, it is assumed to be in the same unit system
            as the potential.
        t : numeric, `~astropy.units.Quantity`, optional
            Time at which to evaluate the circular velocity. Default is 0.

        Returns
        -------
        vcirc : `~astropy.units.Quantity`
            Circular velocity at the spherical radius corresponding to each
            position. For input shape ``(n_dim, n_positions)``, returns shape
            ``(n_positions,)``. Units are velocity (e.g., m/s in SI units).

        Notes
        -----
        This method assumes the potential is approximately spherically
        symmetric. The circular velocity is computed using the relation:

        .. math::
            v_{\\rm circ}(r) = \\sqrt{r \\left| \\frac{d\\Phi}{dr} \\right|}

        where the radial derivative is computed from the Cartesian gradient
        via :math:`dΦ/dr = \\vec{\\nabla}Φ \\cdot \\hat{r}`.

        For exactly spherical potentials, this gives the speed of circular
        orbits. For non-spherical potentials, this provides an approximation
        useful for initial orbit estimates.
        """
        q = self._remove_units_prepare_shape(q)

        # Radius
        r = np.sqrt(np.sum(q**2, axis=0))
        dPhi_dxyz = self.gradient(q, t=t)
        dPhi_dr = np.sum(dPhi_dxyz.value * q / r, axis=0)

        return self._reapply_units_and_shape(
            np.sqrt(r * np.abs(dPhi_dr)),
            self.units[u.get_physical_type("length")]
            / self.units[u.get_physical_type("time")],
            r.shape,
            conv_unit=self.units[u.get_physical_type("velocity")],
        )

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
        subplots_kw=None,
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
        if subplots_kw is None:
            subplots_kw = {}
        grids = []
        slices = []
        for ii, g in enumerate(grid):
            if isiterable(g):
                grids.append((ii, g))
            else:
                slices.append((ii, g))

        # figure out the dimensionality
        ndim = len(grids)

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
            x1 = grids[0][1]
            r = np.zeros((len(grids) + len(slices), len(x1)))
            r[grids[0][0]] = x1

            for ii, slc in slices:
                r[ii] = slc

            Z = self.energy(r * self.units["length"], t=t).value
            ax.plot(x1, Z, **kwargs)

            if labels is not None:
                ax.set_xlabel(labels[0])
                ax.set_ylabel("potential")
        else:
            # 2D contours
            x1, x2 = np.meshgrid(grids[0][1], grids[1][1])
            shp = x1.shape
            x1, x2 = x1.ravel(), x2.ravel()

            r = np.zeros((len(grids) + len(slices), len(x1)))
            r[grids[0][0]] = x1
            r[grids[1][0]] = x2

            for ii, slc in slices:
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
        subplots_kw=None,
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
        if subplots_kw is None:
            subplots_kw = {}
        grids = []
        slices = []
        for ii, g in enumerate(grid):
            if isiterable(g):
                grids.append((ii, g))
            else:
                slices.append((ii, g))

        # figure out the dimensionality
        ndim = len(grids)

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
            x1 = grids[0][1]
            r = np.zeros((len(grids) + len(slices), len(x1)))
            r[grids[0][0]] = x1

            for ii, slc in slices:
                r[ii] = slc

            Z = self.density(r * self.units["length"], t=t).value
            ax.plot(x1, Z, **kwargs)

            if labels is not None:
                ax.set_xlabel(labels[0])
                ax.set_ylabel("potential")
        else:
            # 2D contours
            x1, x2 = np.meshgrid(grids[0][1], grids[1][1])
            shp = x1.shape
            x1, x2 = x1.ravel(), x2.ravel()

            r = np.zeros((len(grids) + len(slices), len(x1)))
            r[grids[0][0]] = x1
            r[grids[1][0]] = x2

            for ii, slc in slices:
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
            R_grid *= self.units["length"]

        xyz = np.zeros((3, *R_grid.shape)) * self.units["length"]
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
        save_all : bool (optional)
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
        pot = pycopy.deepcopy(self) if copy else self

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
        __doc__ = self.energy.__doc__  # noqa: F841
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
        if package == "agama":
            import agama

            from .interop import gala_to_agama_potential

            agama_pot = gala_to_agama_potential(self, **kwargs)
            if not isinstance(agama_pot, agama.Potential):
                agama_pot = agama.Potential(*agama_pot)
            return agama_pot
        raise ValueError(f"Unsupported package: {package}")


class CompositePotential(PotentialBase, OrderedDict):
    """
    A gravitational potential composed of multiple distinct components.

    This class allows combining multiple gravitational potential models
    into a single potential. This is useful for modeling complex systems
    like galaxies, where you might combine a disk, bulge, and dark matter
    halo, each represented by different potential models.

    The `CompositePotential` behaves like a Python dictionary where each
    key-value pair represents a named component and its potential model.
    All components must have compatible unit systems and the same number
    of spatial dimensions.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments where each key is a component name (string) and
        each value is a `~gala.potential.PotentialBase` instance.

    Attributes
    ----------
    lock : bool
        If ``True``, prevents adding new components or modifying existing ones.

    Examples
    --------
    Create a composite potential with named components::

        >>> import astropy.units as u
        >>> from gala.potential import HernquistPotential, NFWPotential
        >>> from gala.units import galactic
        >>> bulge = HernquistPotential(m=1E10*u.Msun, c=1*u.kpc, units=galactic)
        >>> halo = NFWPotential(m=1E12*u.Msun, r_s=20*u.kpc, units=galactic)
        >>> mw = CompositePotential(bulge=bulge, halo=halo)

    Or build it step by step to preserve component order::

        >>> mw = CompositePotential()
        >>> mw['bulge'] = bulge
        >>> mw['halo'] = halo

    Access individual components::

        >>> bulge_potential = mw['bulge']
        >>> total_energy = mw.energy(pos)  # Sum of all components

    Notes
    -----
    The potential energy, gradients, and other quantities are computed as
    the sum over all components. Each component maintains its own parameters
    and can be accessed or modified independently (unless ``lock=True``).

    All components must have the same unit system and spatial dimensionality.
    The composite potential inherits these properties from its components.
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
        super().__setitem__(key, value)

    def _check_component(self, p):
        if not isinstance(p, PotentialBase):
            msg = f"Potential components may only be Potential objects, not {type(p)}."
            raise TypeError(msg)

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
                msg = (
                    "All potential components must have the same "
                    f"number of phase-space dimensions ({self.ndim} in this "
                    "case)"
                )
                raise ValueError(msg)

        if self.lock:
            raise ValueError(
                "Potential object is locked - new components can "
                "only be added to unlocked potentials."
            )

    @property
    def parameters(self):
        params = {}
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
        lock = self.lock
        pots = self.__class__()

        pots._units = None
        pots.lock = False

        for k, v in self.items():
            pots[k] = v.replace_units(units)

        pots.lock = lock
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
