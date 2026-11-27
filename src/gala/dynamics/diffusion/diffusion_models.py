"""Diffusion models for the in-integrator SDE kick; see `DiffusionBase`."""

import functools
import operator

import astropy.units as u
import numpy as np

from ...units import UnitSystem
from .cydiffusion import (
    BASIS_CARTESIAN,
    BASIS_CYLINDRICAL,
    NFIELDS,
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
            "and velocity units). See the DiffusionBase docstring for units."
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
        self.units = units
        self.kick_every = operator.index(kick_every)
        if self.kick_every < 1:
            raise ValueError(f"kick_every must be >= 1, got {kick_every}")
        self.c_instance = None  # subclasses build this

    def kick_capsule(self, kick_seed=None, units=None):
        """A fresh kick (own RNG) for one integration; it keeps this model alive.

        ``kick_seed`` is hashed with `numpy.random.SeedSequence` (``None`` draws fresh
        entropy). If ``units`` is given, its length and time units must match the
        model's.
        """
        if (
            units is not None
            and self.units is not None
            and any(self.units[k] != units[k] for k in ("length", "time"))
        ):
            raise ValueError(
                "diffusion.units must have the same length and time units as "
                "the integration unit system."
            )
        # hashing avoids GSL's aliasing of small seeds (e.g. 0 and 4357). Note
        # SeedSequence ignores trailing zeros, so kick_seed=s and (s, 0) are the same.
        seed = np.random.SeedSequence(kick_seed).generate_state(1, np.uint64)[0]
        return self.c_instance.kick_capsule(int(seed), self.kick_every)

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

    def __reduce__(self):
        # the C model holds raw pointers, so rebuild from the constructor arguments
        # (each subclass defines _init_kwargs)
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
    See `DiffusionBase` for units.
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

        params = np.concatenate([mu, D.ravel()])
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
    See `DiffusionBase` for units.

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
    PSD (e.g. where a coefficient varies sharply). The kick then uses the nearest PSD
    tensor (negative eigenvalues set to zero), which differs from the interpolated one
    by at most the size of its most negative eigenvalue.
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
            fields[:6] = np.moveaxis(drift_grid, -1, 0)
        fields[6:] = np.moveaxis(D_grid[..., *np.triu_indices(6)], -1, 0)

        self.R_grid, self.z_grid, self.D_grid = R, z, D_grid
        self.drift_grid = drift_grid
        fields_flat = fields.ravel()

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


class ISMDiffusionWinter27(DiffusionBase):
    """ISM-driven orbital diffusion and drift of Winter-Granic et al. (2027).

    The model is specified in action space and mapped to position-velocity drift and
    diffusion along each orbit, so it is evaluated per orbit at every kick:

    1. Epicyclic actions ``(J_R, J_phi, J_z)``, the guiding radius, and the frequencies
       ``Omega, kappa, nu`` (with ``q_R = -dln kappa/dln R``) are computed from
       rotation-curve tables built from ``potential``.
    2. Action-space diffusion coefficients are the solar-neighbourhood fits of
       Modak et al. (2026, and in prep.) for ``D_JRJR``, ``D_JphiJphi`` and
       ``D_JzJz`` (cross terms are zero), scaled to each orbit's environment as in
       Winter-Granic et al. (2027, Section 3.1):

       - by ``kappa(R_g)``, ``nu(R_g)`` and ``R_g`` (relative to 40 km/s/kpc,
         92 km/s/kpc and 8 kpc) and by ``(Sigma(R_g) / 12 Msun/pc^2)^2``;
       - by ``tau_star / 70 Myr`` and a power of ``lambda_star / 600 pc``: 2.3 in the
         long-wavelength regime, and 3.3 or 5.3 in the short-wavelength regimes;
       - for ``D_JRJR`` and ``D_JphiJphi``, by a planar-suppression factor ``xi`` for
         vertically hot orbits, which depends on ``J_z``, ``nu`` and ``H_gas``
         (Modak et al. 2026, eq. 48);
       - for ``D_JzJz``, by ``((1 + 2 pi H_gas / lambda_star) / 3.1)^-2``.

       Each coefficient is a broken power law in the actions that is continuous at
       its breaks, which therefore move with the scalings (at the fiducial values they
       are at ``J_R`` = 0.73, 2.4 and 1.4 kpc km/s for ``D_JphiJphi``, ``D_JRJR`` and
       ``D_JzJz``, and ``J_z`` = 1.0 kpc km/s).
    3. Drifts are ``A_i = 1/2 dD_ii/dJ_i``. For ``J_phi`` this acts through the
       guiding radius: ``A_Jphi = Omega / (R_g kappa^2) dD_JphiJphi/dR_g``.
    4. ``(A_J, D_J)`` are mapped to cylindrical ``(R, phi, z, v_R, v_phi, v_z)`` drift
       and diffusion with the paper's epicyclic formulas (Section 2.3 and App. A).

    Parameters
    ----------
    potential : `~gala.potential.PotentialBase`
        Axisymmetric potential used to tabulate the rotation curve (along the x axis
        in the z=0 plane). The model uses ``potential.units``, which must match the
        integration Hamiltonian's length and time units.
    amplitude : float, optional
        Strength of the ISM density fluctuations relative to the fiducial model; every
        diffusion and drift coefficient scales as ``amplitude**2``. Default 1.
    Sigma : `~astropy.units.Quantity` or callable, optional
        Mean gas surface density of the ISM: a constant, or a function of cylindrical
        radius (a Quantity) that returns a surface density Quantity. It is evaluated at
        each orbit's guiding radius. Default 12 Msun/pc^2.
    lambda_star : `~astropy.units.Quantity`, optional
        Dominant spatial scale of the ISM fluctuations. Besides rescaling the fits, it
        sets the wavenumber ``k_* = 2 pi / lambda_star`` of the planar suppression.
        Default 600 pc.
    tau_star : `~astropy.units.Quantity`, optional
        Correlation time of the ISM fluctuations at ``lambda_star``. Default 70 Myr.
    H_gas : `~astropy.units.Quantity`, optional
        Thickness of the gas layer, which sets the planar suppression of vertically
        hot orbits and the vertical coefficient. Default 200 pc.
    long_wavelength_only : bool, optional
        Always use the long-wavelength (low-action) branches of the fits, as in the
        paper's Gaia-snail and bar-resonance examples. Default False.
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
    - The long-radial-wavelength, short-vertical-wavelength regime keeps the
      long-wavelength vertical coefficient, as in the paper. ``D_JzJz`` is therefore
      discontinuous where a vertically hot orbit (``J_z`` above its break) crosses the
      radial break; elsewhere all coefficients are continuous.
    - ``A_JR`` and ``A_Jz`` carry the same environmental factors as ``D_JRJR`` and
      ``D_JzJz``, so ``A = 1/2 dD/dJ`` holds throughout.
    """

    def __init__(
        self,
        potential,
        amplitude=1.0,
        Sigma=12 * u.Msun / u.pc**2,
        lambda_star=600 * u.pc,
        tau_star=70 * u.Myr,
        H_gas=200 * u.pc,
        long_wavelength_only=False,
        R_min=0.5 * u.kpc,
        R_max=50.0 * u.kpc,
        n_R=512,
        J_floor=1e-3 * u.kpc * u.km / u.s,
        kick_every=1,
    ):
        self._kwargs = {
            "potential": potential,
            "amplitude": amplitude,
            "Sigma": Sigma,
            "lambda_star": lambda_star,
            "tau_star": tau_star,
            "H_gas": H_gas,
            "long_wavelength_only": long_wavelength_only,
            "R_min": R_min,
            "R_max": R_max,
            "n_R": n_R,
            "J_floor": J_floor,
        }
        ratios = {}
        for name, value, fiducial in [
            ("lambda_star", lambda_star, 600 * u.pc),
            ("tau_star", tau_star, 70 * u.Myr),
            ("H_gas", H_gas, 200 * u.pc),
        ]:
            ratio = u.Quantity(value).to_value(fiducial.unit)
            if not (np.isscalar(ratio) and np.isfinite(ratio) and ratio > 0):
                raise ValueError(f"{name} must be a positive scalar Quantity.")
            ratios[name] = ratio / fiducial.value
        tau, lam = ratios["tau_star"], ratios["lambda_star"]

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

        Sigma_R = Sigma(R * units["length"]) if callable(Sigma) else Sigma
        Sigma_R = np.broadcast_to(
            u.Quantity(Sigma_R).to_value(u.Msun / u.pc**2) / 12.0, R.shape
        )
        if not np.all(np.isfinite(Sigma_R) & (Sigma_R > 0)):
            raise ValueError("Sigma must be positive and finite over [R_min, R_max].")

        def val(x):
            return x.decompose(units).value

        params = np.array(
            [
                float(amplitude) ** 2,
                val(1 * u.kpc * u.km / u.s),
                val(1 * (u.kpc * u.km / u.s) ** 2 / u.Gyr),
                val(J_floor),
                float(long_wavelength_only),
                val(40.0 * u.km / u.s / u.kpc),
                val(92.0 * u.km / u.s / u.kpc),
                val(8.0 * u.kpc),
                val(H_gas),
                val(2 * np.pi / lambda_star),
                tau * lam**2.3,
                tau * lam**3.3,
                tau * lam**5.3,
                ((1 + (2 * np.pi * H_gas / lambda_star).to_value(u.one)) / 3.1) ** -2,
            ]
        )

        wrapper = CDiffusionWrapper()
        wrapper.init_winter27(
            np.log(R),
            0.5 * np.log(Omega2),
            0.5 * np.log(kappa2),
            0.5 * np.log(nu2),
            np.log(Sigma_R),
            lnLz,
            params,
        )
        self.c_instance = wrapper

    def _init_kwargs(self):
        return {**self._kwargs, "kick_every": self.kick_every}
