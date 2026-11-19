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

import operator

import astropy.units as u
import numpy as np

from ...units import UnitSystem
from .cydiffusion import (
    BASIS_CARTESIAN,
    BASIS_CYLINDRICAL,
    CDiffusionWrapper,
)

__all__ = [
    "ConstantDiffusion",
    "DiffusionBase",
    "GriddedDiffusion",
    "ISMDiffusionWinter27",
]

_BASIS = {"cartesian": BASIS_CARTESIAN, "cylindrical": BASIS_CYLINDRICAL}


def _check_plain(name, arr):
    if hasattr(arr, "unit"):
        raise TypeError(
            f"`{name}` must be a plain numeric array in the model's unit system, "
            "not an astropy Quantity (the full phase-space tensor mixes position "
            "and velocity units). See the gala.dynamics.diffusion unit convention."
        )
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
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
    if np.any(d < 0) or np.linalg.eigvalsh(C).min() < -tol:
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
    already be in that system. For ``basis='cylindrical'`` the components are
    physical values along the (R-hat, phi-hat, z-hat) unit vectors -- the ``phi``
    entries are lengths / velocities, not radians.
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
        xv = np.ascontiguousarray(np.asarray(w, dtype=np.float64).reshape(6, -1).T)
        return self.c_instance.coefficients(xv, float(t))


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
    drift, no diffusion) -- values are not extrapolated beyond the grid.

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


class ISMDiffusionWinter27(DiffusionBase):
    """ISM-driven orbital diffusion and drift of Winter-Granic et al. (2027).

    The model is specified in action space and mapped to position-velocity drift and
    diffusion along each orbit, so it is evaluated per orbit at every kick:

    1. Epicyclic actions ``(J_R, J_phi, J_z)``, the guiding radius, and the frequencies
       ``Omega, kappa, nu`` (with ``q_R = -dln kappa/dln R``) are computed from
       rotation-curve tables built from ``potential``.
    2. Action-space diffusion coefficients are the Modak et al. (2026) fits for
       ``D_JRJR``, ``D_JphiJphi``, and ``D_JzJz``. The short- and long-wavelength
       regimes switch at fixed actions (2.4, 0.72, and 1.5 kpc km/s in ``J_R`` for the
       radial, azimuthal and vertical coefficients; 2.3 kpc km/s in ``J_z``). Cross
       terms are zero.
    3. Drifts are the quasilinear ones, ``A_i = 1/2 dD_ii/dJ_i``. ``A_Jphi = 0``.
    4. ``(A_J, D_J)`` are mapped to cylindrical ``(R, phi, z, v_R, v_phi, v_z)`` drift
       and diffusion with the paper's epicyclic formulas (App. B).

    This follows the authors' reference script (``eom_model.py``), including the
    optional corrections below.

    Parameters
    ----------
    potential : `~gala.potential.PotentialBase`
        Axisymmetric potential used to tabulate the rotation curve (along the x axis
        in the z=0 plane). The model uses ``potential.units``, which must match the
        integration Hamiltonian's length and time units.
    scale : float, optional
        Multiplies every diffusion and drift coefficient (the square of the ISM
        fluctuation amplitude). Default 1.
    planar_suppression : bool, optional
        Multiply ``D_JRJR``, ``A_JR`` and ``D_JphiJphi`` by a factor ``xi(J_z)`` that
        accounts for the finite thickness of the gas disc (scale height 250 pc,
        ``k_* = 0.01/pc``): vertically hot orbits spend less time in the gas and feel
        weaker in-plane forcing. Default True.
    environmental_scaling : bool, optional
        The fits are measured in the solar neighbourhood. If True, rescale them to each
        orbit's guiding radius using ``kappa(R_g)``, ``nu(R_g)`` and ``R_g`` (relative
        to ``sqrt(2) 28 km/s/kpc``, ``2 pi / 75.55 Myr`` and 8 kpc), to extrapolate the
        diffusion to other radii. If False, the same rescaling is applied once with the
        potential's values at 8 kpc. Default False.
    long_wavelength_only : bool, optional
        Always use the long-wavelength (low-action) branches of the fits, as in the
        paper's Gaia-snail and bar-resonance examples. Default False.
    vertical : bool, optional
        Include vertical (``J_z``) diffusion and drift. Set False for in-plane-only
        experiments. Default True.
    R_min, R_max : `~astropy.units.Quantity`, optional
        Range of the rotation-curve tables. Orbits whose guiding radius falls outside
        get no kick. Defaults 0.5 and 50 kpc.
    n_R : int, optional
        Number of (log-spaced) table points. Default 512.
    J_floor : `~astropy.units.Quantity`, optional
        Floor on ``J_R`` and ``J_z`` used in the mapping, which is singular as either
        action goes to zero. Default ``1e-3 kpc km/s``.
    kick_every : int, optional
        Apply the kick every ``kick_every`` integrator steps, with the interval since
        the last kick as its timestep. The coefficients vary along the epicycle, so
        keep ``kick_every * dt`` well below the epicyclic period. Default 1.

    Notes
    -----
    - The paper defines ``q_R = dln kappa/dln R`` but states ``q_R = 1`` for a flat
      rotation curve; the minus-sign convention is used here.
    - The short-wavelength ``D_JphiJphi`` normalization is 5900 (kpc km/s)^2/Gyr, as in
      the reference script (the 0.72 kpc km/s break is where it meets the 6900
      plateau); the paper text says 4600.
    - With ``environmental_scaling``, ``D_JphiJphi`` depends on ``J_phi`` through
      ``R_g``, but the drift ``A_Jphi`` is still set to zero, as in the script.
    """

    def __init__(
        self,
        potential,
        scale=1.0,
        planar_suppression=True,
        environmental_scaling=False,
        long_wavelength_only=False,
        vertical=True,
        R_min=0.5 * u.kpc,
        R_max=50.0 * u.kpc,
        n_R=512,
        J_floor=1e-3 * u.kpc * u.km / u.s,
        kick_every=1,
    ):
        units = potential.units
        super().__init__(basis="cylindrical", units=units, kick_every=kick_every)
        self.potential = potential

        R = np.geomspace(
            R_min.decompose(units).value, R_max.decompose(units).value, int(n_R)
        )
        q = np.zeros((3, R.size))
        q[0] = R
        q = q * units["length"]
        Phi_R = potential.gradient(q).decompose(units).value[0]
        hess = potential.hessian(q).decompose(units).value
        Omega2 = Phi_R / R
        kappa2 = hess[0, 0] + 3 * Phi_R / R
        nu2 = hess[2, 2]
        if np.any(Omega2 <= 0) or np.any(kappa2 <= 0) or np.any(nu2 <= 0):
            raise ValueError(
                "The potential must have positive Omega^2, kappa^2, and nu^2 over "
                "[R_min, R_max] in the z=0 plane."
            )
        lnLz = np.log(R**2 * np.sqrt(Omega2))
        if np.any(np.diff(lnLz) <= 0):
            raise ValueError(
                "L_z = R v_c(R) must increase with R over [R_min, R_max] to define "
                "guiding radii."
            )

        def val(x):
            return x.decompose(units).value

        R_ref = val(8.0 * u.kpc)
        lnR = np.log(R)
        kappa_8 = np.exp(np.interp(np.log(R_ref), lnR, 0.5 * np.log(kappa2)))
        nu_8 = np.exp(np.interp(np.log(R_ref), lnR, 0.5 * np.log(nu2)))
        params = np.array(
            [
                float(scale),
                val(1 * u.kpc * u.km / u.s),
                val(1 * (u.kpc * u.km / u.s) ** 2 / u.Gyr),
                val(J_floor),
                float(planar_suppression),
                float(environmental_scaling),
                float(long_wavelength_only),
                float(vertical),
                val(np.sqrt(2) * 28 * u.km / u.s / u.kpc),
                val(2 * np.pi / (75.55 * u.Myr)),
                R_ref,
                kappa_8,
                nu_8,
                val(250 * u.pc),
                val(0.01 / u.pc),
            ]
        )

        wrapper = CDiffusionWrapper()
        wrapper.init_winter27(
            lnR,
            0.5 * np.log(Omega2),
            0.5 * np.log(kappa2),
            0.5 * np.log(nu2),
            lnLz,
            params,
        )
        self.c_instance = wrapper
