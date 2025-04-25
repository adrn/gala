import numpy as np
from scipy.linalg import svd

from .actionangle_o2gf import _action_prepare, _angle_prepare, fit_toy_potential


def solve_low_rank(A, b, rank=None, tol=1e-10):
    """
    Solve Ax = b using a low-rank approximation to A via truncated SVD.

    Parameters
    ----------
    A : array_like
    b : array_like
    rank : int, optional
        Number of singular values to keep. If None, use all above `tol`.
    tol : float
        Threshold below which singular values are discarded.

    Returns
    -------
    x : array_like
        Solution vector.
    """
    U, s, Vh = svd(A)
    if rank is None:
        rank = np.sum(s > tol)
    S_inv = np.zeros_like(s)
    S_inv[:rank] = 1.0 / s[:rank]
    A_pinv = (Vh.T @ np.diag(S_inv)) @ U.T
    return A_pinv @ b


def _single_orbit_find_actions_regularized(
    orbit,
    N_max,
    rank=None,
    tol=1e-10,
    toy_potential=None,
    force_harmonic_oscillator=False,
    fit_kwargs=None,
):
    import warnings

    from gala.logging import logger
    from gala.potential import HarmonicOscillatorPotential, IsochronePotential

    if orbit.norbits > 1:
        raise ValueError("must be a single orbit")

    if fit_kwargs is None:
        fit_kwargs = {}

    if toy_potential is None:
        toy_potential = fit_toy_potential(
            orbit, force_harmonic_oscillator=force_harmonic_oscillator, **fit_kwargs
        )
    else:
        logger.debug(f"Using *fixed* toy potential: {toy_potential.parameters}")

    if isinstance(toy_potential, IsochronePotential):
        orbit_align = orbit.align_circulation_with_z()
        w = orbit_align.w()

        dxyz = (1, 2, 2)
        circ = np.sign(w[0, 0] * w[4, 0] - w[1, 0] * w[3, 0])
        sign = np.array([1.0, circ, 1.0])
        orbit = orbit_align
    elif isinstance(toy_potential, HarmonicOscillatorPotential):
        dxyz = (2, 2, 2)
        sign = 1.0
        w = orbit.w()
    else:
        raise ValueError("Invalid toy potential.")

    t = orbit.t.value
    aaf = toy_potential.action_angle(orbit)
    if aaf[0].ndim > 2:
        aa = np.vstack((aaf[0].value[..., 0], aaf[1].value[..., 0]))
    else:
        aa = np.vstack((aaf[0].value, aaf[1].value))

    if np.any(np.isnan(aa)):
        ix = ~np.any(np.isnan(aa), axis=0)
        aa = aa[:, ix]
        t = t[ix]
        warnings.warn("NaN value in toy actions or angles!")
        if sum(ix) > 1:
            raise ValueError("Too many NaN value in toy actions or angles!")

    J_toy = aa[:3]  # shape (3, N)
    J_mean = np.mean(J_toy, axis=1, keepdims=True)
    aa[:3] -= J_mean

    A, b, nvecs = _action_prepare(aa, N_max, dx=dxyz[0], dy=dxyz[1], dz=dxyz[2])
    actions = solve_low_rank(A, b, rank=rank, tol=tol)

    A, b, nvecs = _angle_prepare(
        aa, t, N_max, dx=dxyz[0], dy=dxyz[1], dz=dxyz[2], sign=sign
    )
    angles = solve_low_rank(A, b, rank=rank, tol=tol)

    J = actions[:3] + J_mean
    theta = angles[:3]
    freqs = angles[3:6]

    return dict(
        actions=J * aaf[0].unit,
        angles=theta * aaf[1].unit,
        freqs=freqs * aaf[2].unit,
        Sn=actions[3:],
        dSn_dJ=angles[6:],
        nvecs=nvecs,
    )


def find_actions_o2gf_regularized(
    orbit,
    N_max,
    rank=None,
    tol=1e-10,
    force_harmonic_oscillator=False,
    toy_potential=None,
    fit_kwargs=None,
):
    from astropy.table import QTable

    if orbit.norbits == 1:
        result = _single_orbit_find_actions_regularized(
            orbit,
            N_max,
            rank=rank,
            tol=tol,
            force_harmonic_oscillator=force_harmonic_oscillator,
            toy_potential=toy_potential,
            fit_kwargs=fit_kwargs,
        )
        rows = [result]
    else:
        rows = []
        for n in range(orbit.norbits):
            aaf = _single_orbit_find_actions_regularized(
                orbit[:, n],
                N_max,
                rank=rank,
                tol=tol,
                force_harmonic_oscillator=force_harmonic_oscillator,
                toy_potential=toy_potential,
                fit_kwargs=fit_kwargs,
            )
            rows.append(aaf)

    return QTable(rows=rows)
