"""
Python-facing diffusion models for the trial in-integrator SDE kick.

A model provides a drift vector ``mu`` (length 6) and a diffusion tensor ``D`` (6x6,
symmetric) over the phase space ``(x, y, z, vx, vy, vz)``, expressed in a chosen
``basis`` and either constant or varying over a ``(R, |z|)`` grid. The per-step kick
has increment ``dw = mu*dt + L*sqrt(dt)*xi`` with ``L L^T = D`` (rotated into
Cartesian), i.e. ``Cov(dw) = D*dt``.

Unit convention
---------------
Coefficients are plain numeric arrays already expressed in the integrator's unit
system, with phase-space ordering ``(R, phi, z, v_R, v_phi, v_z)`` for cylindrical
basis (or ``(x, y, z, vx, vy, vz)`` for cartesian). A single astropy Quantity cannot
carry the mixed position/velocity units of the full tensor, so Quantities are rejected
-- pass numbers in the system's length and length/time units (e.g. galactic: kpc,
kpc/Myr, Myr). Build the model in the same unit system you integrate in.
"""

import functools
import operator

import numpy as np

from ...units import UnitSystem
from .cydiffusion import (
    BASIS_CARTESIAN,
    BASIS_CYLINDRICAL,
    NFIELDS,
    CDiffusionWrapper,
)

__all__ = ["ConstantDiffusion", "DiffusionBase", "GriddedDiffusion"]

_BASIS = {"cartesian": BASIS_CARTESIAN, "cylindrical": BASIS_CYLINDRICAL}


def _check_plain(name, arr):
    if hasattr(arr, "unit"):
        raise TypeError(
            f"`{name}` must be a plain numeric array in the model's unit system, "
            "not an astropy Quantity (the full phase-space tensor mixes position "
            "and velocity units). See the gala.dynamics.diffusion unit convention."
        )
    arr = np.array(arr, dtype=np.float64, order="C")  # a copy: don't alias the input
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"`{name}` must be finite (no NaN or inf).")
    return arr


def _check_psd(name, D, tol=1e-10):
    """Check that the trailing (6, 6) tensor(s) in D are symmetric and PSD.

    The checks are done on D normalized by sqrt(D_ii D_jj): blocks have mixed units and
    realistic coefficients can be ~1e-9 in natural units, so absolute (or max-relative)
    tolerances would accept almost anything.
    """
    d = np.diagonal(D, axis1=-2, axis2=-1)
    s = np.sqrt(np.abs(d))
    s[s == 0] = 1.0
    C = D / (s[..., :, None] * s[..., None, :])
    if np.abs(C - np.swapaxes(C, -1, -2)).max() > tol:
        raise ValueError(f"`{name}` must be symmetric.")
    # a zero diagonal forces its whole row/column to zero (otherwise D is indefinite,
    # however small the off-diagonal entry; the normalization above can't see it)
    zero = d == 0
    if (
        np.any(d < 0)
        or np.any(zero[..., :, None] & (D != 0))
        or np.linalg.eigvalsh(C).min() < -tol
    ):
        raise ValueError(f"`{name}` must be positive semi-definite.")


class DiffusionBase:
    """Base class for diffusion models used by the in-integrator kick.

    Notes
    -----
    **Units subtlety.** Coefficients must be passed as plain numeric arrays already
    expressed in the base units of ``units`` -- astropy ``Quantity`` inputs are
    rejected with a ``TypeError``. The reason is that the diffusion tensor spans the
    full phase space and its blocks have *different* physical dimensions, so a single
    ``Quantity`` (one unit) cannot represent it. With the convention
    ``Cov(dw) = D * dt`` (and ``dw = mu * dt`` for the drift), for phase-space
    ordering ``(x, y, z, vx, vy, vz)`` the required units are:

    - position-position block ``D[0:3, 0:3]``: ``length**2 / time``
    - velocity-velocity block ``D[3:6, 3:6]``: ``velocity**2 / time``
    - position-velocity block ``D[0:3, 3:6]``: ``length * velocity / time``
    - drift ``mu[0:3]`` (positions): ``length / time``;
      ``mu[3:6]`` (velocities): ``velocity / time``

    For example, in ``galactic`` units (kpc, Myr) a velocity-diffusion rate is in
    ``(kpc/Myr)**2 / Myr``. Convert from your preferred units with astropy and pass
    the plain value, e.g.::

        D[3, 3] = (1500 * (u.km / u.s) ** 2 / u.Gyr).to_value(
            (u.kpc / u.Myr) ** 2 / u.Myr
        )

    ``units`` is only used to check consistency against the integration Hamiltonian
    (a mismatch raises); it does **not** convert the coefficients, so they must
    already be in that system.

    **Cylindrical basis.** With ``basis='cylindrical'`` the model is an Ito SDE in the
    cylindrical coordinates ``(R, phi, z, v_R, v_phi, v_z)``: each kick updates those
    coordinates directly, and the velocities stay attached to the local frame at the
    new azimuth. The ``phi`` entries are in arc length (``R dphi``) and the
    corresponding velocity, not radians. This is the natural form for models built
    from actions/angles, but note that it is *not* the same as an isotropic physical
    displacement: an azimuthal position kick moves a star along its circle and leaves
    ``R`` unchanged, so the curvature drift ``D_phiphi / (2 R)`` in ``R`` that a
    straight-line kick would produce is not added. For isotropic *position* diffusion
    use ``basis='cartesian'`` (or add that drift yourself). Velocity-only diffusion is
    the same in either basis.

    **Seeds.** The kick's random number generator (GSL ``mt19937``) uses 32 bits of
    the hashed ``kick_seed``, so distinct seeds collide with probability ~2^-32 per
    pair.
    """

    def __init__(self, basis="cylindrical", units=None, kick_every=1):
        if basis not in _BASIS:
            raise ValueError(f"Unknown basis '{basis}'. Options: {sorted(_BASIS)}")
        self.basis = basis
        if units is not None and not isinstance(units, UnitSystem):
            units = UnitSystem(units)
        self._units = units
        self.kick_every = operator.index(kick_every)
        if self.kick_every < 1:
            raise ValueError(f"kick_every must be >= 1, got {kick_every}")
        self.c_instance = None  # subclasses build this

    @property
    def units(self):
        return self._units

    def kick_capsule(self, seed):
        """A fresh kick (own RNG) for one integration; it keeps this model alive."""
        return self.c_instance.kick_capsule(seed, self.kick_every)

    def coefficients(self, w, t=0.0):
        """Evaluate the model's drift and diffusion tensor at phase-space points.

        Parameters
        ----------
        w : `~gala.dynamics.PhaseSpacePosition` or array_like
            Positions, or a ``(6, n)`` array of Cartesian ``(x, y, z, vx, vy, vz)`` in
            the model's units.
        t : float, optional
            Time, in the model's time unit.

        Returns
        -------
        mu : ndarray, shape ``(n, 6)``
            Drift, in the model's ``basis``.
        D : ndarray, shape ``(n, 6, 6)``
            Diffusion tensor, in the model's ``basis``.
        """
        if hasattr(w, "w"):
            w = w.w(self.units)
        w = np.asarray(w, dtype=np.float64)
        if w.ndim > 2 or w.shape[0] != 6:
            raise ValueError(
                f"`w` must have shape (6,) or (6, n) (x, y, z, vx, vy, vz), got {w.shape}"
            )
        xv = np.ascontiguousarray(w.reshape(6, -1).T)
        return self.c_instance.coefficients(xv, float(t))

    def _init_kwargs(self):
        """Constructor arguments that rebuild this model (used for pickling)."""
        raise NotImplementedError

    def __reduce__(self):
        # the C model holds raw pointers, so rebuild from the constructor arguments
        return (functools.partial(type(self), **self._init_kwargs()), ())


class ConstantDiffusion(DiffusionBase):
    """Constant drift + diffusion tensor (independent of position/velocity).

    Parameters
    ----------
    D : (6, 6) array_like
        Symmetric phase-space diffusion tensor; ``Cov(dw) = D*dt``. Plain numeric
        array in the base units of ``units`` (see Notes) -- not a `~astropy.units.Quantity`.
    drift : (6,) array_like, optional
        Deterministic phase-space drift ``mu`` (e.g. dynamical friction), ``dw = mu*dt``.
        Plain numeric array in the base units of ``units`` (see Notes). Default 0.
    basis : str
        'cylindrical' (default) or 'cartesian'.
    units : `~gala.units.UnitSystem`, optional
        Unit system the coefficients are expressed in.
    kick_every : int, optional
        Apply the kick every ``kick_every`` integrator steps, with the interval since
        the last kick as its timestep. Default 1 (every step).

    Notes
    -----
    ``D`` and ``drift`` must be plain numeric arrays already in the base units of
    ``units``; astropy ``Quantity`` inputs are rejected because the full 6x6 tensor
    mixes position and velocity units. See `DiffusionBase` for the per-block unit
    table and a conversion example.
    """

    def __init__(self, D, drift=None, basis="cylindrical", units=None, kick_every=1):
        super().__init__(basis=basis, units=units, kick_every=kick_every)

        D = _check_plain("D", D)
        if D.shape != (6, 6):
            raise ValueError(f"`D` must have shape (6, 6), got {D.shape}")
        _check_psd("D", D)

        if drift is None:
            mu = np.zeros(6)
        else:
            mu = _check_plain("drift", drift).ravel()
            if mu.shape != (6,):
                raise ValueError(f"`drift` must have shape (6,), got {mu.shape}")

        self.D = D
        self.drift = mu

        params = np.ascontiguousarray(np.concatenate([mu, D.ravel()]), dtype=np.float64)
        wrapper = CDiffusionWrapper()
        wrapper.init_constant(params, _BASIS[basis])
        self.c_instance = wrapper

    def _init_kwargs(self):
        return {
            "D": self.D,
            "drift": self.drift,
            "basis": self.basis,
            "units": self.units,
            "kick_every": self.kick_every,
        }


class GriddedDiffusion(DiffusionBase):
    """Drift + diffusion tensor interpolated over a regular ``(R, |z|)`` grid.

    Fill the grid by evaluating your own function of galactocentric cylindrical
    ``R`` and ``|z|``; it is interpolated in C (bicubic ``gsl_spline2d``) at each step.

    Parameters
    ----------
    R_grid : (nR,) array_like
        Monotonic cylindrical radius grid (nR >= 4), in the length unit of ``units``.
    z_grid : (nz,) array_like
        Monotonic ``|z|`` grid (nz >= 4), in the length unit of ``units``.
    D_grid : (nR, nz, 6, 6) array_like
        Symmetric diffusion tensor at each grid node; ``Cov(dw) = D*dt``.
    drift_grid : (nR, nz, 6) array_like, optional
        Drift vector ``mu`` at each grid node, ``dw = mu*dt``. Default 0.
    basis : str
        'cylindrical' (default) or 'cartesian'.
    units : `~gala.units.UnitSystem`, optional
        Unit system the coefficients/grid are expressed in.
    kick_every : int, optional
        Apply the kick every ``kick_every`` integrator steps, with the interval since
        the last kick as its timestep. Default 1 (every step).

    Notes
    -----
    All inputs (``R_grid``, ``z_grid``, ``D_grid``, ``drift_grid``) must be plain
    numeric arrays already in the base units of ``units``; astropy ``Quantity``
    inputs are rejected because the full 6x6 tensor mixes position and velocity
    units. See `DiffusionBase` for the per-block unit table (each ``D_grid`` node
    follows the same block-unit convention) and a conversion example.

    Outside the ``(R, |z|)`` grid bounds the coefficients are taken to be zero (no
    drift, no diffusion) -- values are not extrapolated beyond the grid. So a grid
    that starts above ``|z| = 0`` (or above the smallest ``R`` your orbits reach)
    switches diffusion off there, e.g. at the midplane; start ``z_grid`` at 0 to
    cover it.

    The grid holds values for ``z >= 0`` and the model is assumed mirror-symmetric
    about the midplane: at ``z < 0`` the components that are odd under
    ``z -> -z`` change sign. These are the drifts ``mu_z`` and ``mu_vz`` and the
    tensor entries pairing exactly one of ``z, v_z`` with an in-plane component
    (e.g. ``D[v_R, v_z]``, the tilt of the velocity ellipsoid).

    Each tensor component is interpolated independently, so between nodes the
    interpolated ``D`` can be slightly non-positive-definite even when every node is
    PSD (e.g. where a coefficient varies sharply). Any negative directions are then
    clamped to zero by the kick's Cholesky factorization.
    """

    def __init__(
        self,
        R_grid,
        z_grid,
        D_grid,
        drift_grid=None,
        basis="cylindrical",
        units=None,
        kick_every=1,
    ):
        super().__init__(basis=basis, units=units, kick_every=kick_every)

        R = _check_plain("R_grid", R_grid).ravel()
        z = _check_plain("z_grid", z_grid).ravel()
        nR, nz = R.shape[0], z.shape[0]
        if nR < 4 or nz < 4:
            raise ValueError(
                "R_grid and z_grid must each have >= 4 points (bicubic interpolation)."
            )
        if np.any(np.diff(R) <= 0) or np.any(np.diff(z) <= 0):
            raise ValueError("R_grid and z_grid must be strictly increasing.")
        if z[0] < 0:
            raise ValueError("z_grid is a grid in |z| and must be >= 0.")

        D_grid = _check_plain("D_grid", D_grid)
        if D_grid.shape != (nR, nz, 6, 6):
            raise ValueError(
                f"`D_grid` must have shape ({nR}, {nz}, 6, 6), got {D_grid.shape}"
            )
        _check_psd("D_grid", D_grid)

        # pack fields: [0:6] = drift, [6:27] = upper-triangular tensor
        fields = np.zeros((NFIELDS, nR, nz), dtype=np.float64)
        if drift_grid is not None:
            drift_grid = _check_plain("drift_grid", drift_grid)
            if drift_grid.shape != (nR, nz, 6):
                raise ValueError(
                    f"`drift_grid` must have shape ({nR}, {nz}, 6), got "
                    f"{drift_grid.shape}"
                )
            for k in range(6):
                fields[k] = drift_grid[..., k]
        f = 6
        for i in range(6):
            for j in range(i, 6):
                fields[f] = D_grid[..., i, j]
                f += 1

        self.R_grid, self.z_grid, self.D_grid = R, z, D_grid
        self.drift_grid = drift_grid
        fields_flat = np.ascontiguousarray(fields.ravel(), dtype=np.float64)

        wrapper = CDiffusionWrapper()
        wrapper.init_gridded(R, z, fields_flat, nR, nz, _BASIS[basis])
        self.c_instance = wrapper

    def _init_kwargs(self):
        return {
            "R_grid": self.R_grid,
            "z_grid": self.z_grid,
            "D_grid": self.D_grid,
            "drift_grid": self.drift_grid,
            "basis": self.basis,
            "units": self.units,
            "kick_every": self.kick_every,
        }
