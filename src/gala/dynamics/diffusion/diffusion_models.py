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

import numpy as np

from ...units import UnitSystem
from .cydiffusion import (
    BASIS_CARTESIAN,
    BASIS_CYLINDRICAL,
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
    return np.ascontiguousarray(np.asarray(arr, dtype=np.float64))


class DiffusionBase:
    """Base class for diffusion models used by the in-integrator kick."""

    ndim = 3

    def __init__(self, basis="cylindrical", units=None):
        if basis not in _BASIS:
            raise ValueError(f"Unknown basis '{basis}'. Options: {sorted(_BASIS)}")
        self.basis = basis
        if units is not None and not isinstance(units, UnitSystem):
            units = UnitSystem(units)
        self._units = units
        self.c_instance = None  # subclasses build this

    @property
    def units(self):
        return self._units

    def set_seed(self, seed):
        self.c_instance.set_seed(int(seed))

    def kick_capsule(self):
        return self.c_instance.kick_capsule()


class ConstantDiffusion(DiffusionBase):
    """Constant drift + diffusion tensor (independent of position/velocity).

    Parameters
    ----------
    D : (6, 6) array_like
        Symmetric phase-space diffusion tensor; ``Cov(dw) = D*dt``.
    drift : (6,) array_like, optional
        Deterministic phase-space drift ``mu`` (e.g. dynamical friction). Default 0.
    basis : str
        'cylindrical' (default) or 'cartesian'.
    units : `~gala.units.UnitSystem`, optional
        Unit system the coefficients are expressed in.
    """

    def __init__(self, D, drift=None, basis="cylindrical", units=None):
        super().__init__(basis=basis, units=units)

        D = _check_plain("D", D)
        if D.shape != (6, 6):
            raise ValueError(f"`D` must have shape (6, 6), got {D.shape}")
        if not np.allclose(D, D.T):
            raise ValueError("`D` must be symmetric.")

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


class GriddedDiffusion(DiffusionBase):
    """Drift + diffusion tensor interpolated over a regular ``(R, |z|)`` grid.

    Fill the grid by evaluating your own function of galactocentric cylindrical
    ``R`` and ``|z|``; it is interpolated in C (bicubic ``gsl_spline2d``) at each step.

    Parameters
    ----------
    R_grid : (nR,) array_like
        Monotonic cylindrical radius grid (nR >= 4), in system length units.
    z_grid : (nz,) array_like
        Monotonic ``|z|`` grid (nz >= 4), in system length units.
    D_grid : (nR, nz, 6, 6) array_like
        Symmetric diffusion tensor at each grid node.
    drift_grid : (nR, nz, 6) array_like, optional
        Drift vector at each grid node. Default 0.
    basis : str
        'cylindrical' (default) or 'cartesian'.
    units : `~gala.units.UnitSystem`, optional
        Unit system the coefficients/grid are expressed in.
    """

    def __init__(
        self, R_grid, z_grid, D_grid, drift_grid=None, basis="cylindrical", units=None
    ):
        super().__init__(basis=basis, units=units)

        R = _check_plain("R_grid", R_grid).ravel()
        z = _check_plain("z_grid", z_grid).ravel()
        nR, nz = R.shape[0], z.shape[0]
        if nR < 4 or nz < 4:
            raise ValueError(
                "R_grid and z_grid must each have >= 4 points (bicubic interpolation)."
            )
        if np.any(np.diff(R) <= 0) or np.any(np.diff(z) <= 0):
            raise ValueError("R_grid and z_grid must be strictly increasing.")

        D_grid = _check_plain("D_grid", D_grid)
        if D_grid.shape != (nR, nz, 6, 6):
            raise ValueError(
                f"`D_grid` must have shape ({nR}, {nz}, 6, 6), got {D_grid.shape}"
            )

        # pack fields: [0:6] = drift, [6:27] = upper-triangular tensor
        fields = np.zeros((27, nR, nz), dtype=np.float64)
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
        fields_flat = np.ascontiguousarray(fields.ravel(), dtype=np.float64)

        wrapper = CDiffusionWrapper()
        wrapper.init_gridded(R, z, fields_flat, nR, nz, _BASIS[basis])
        self.c_instance = wrapper
