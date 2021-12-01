# coding: utf-8

# Third-party
import numpy as np
import scipy.integrate as si

# Project
from ._computecoeff import (
    Snlm_integrand,
    Tnlm_integrand,
    STnlm_discrete,
    STnlm_var_discrete,
)

__all__ = ["compute_coeffs", "compute_coeffs_discrete"]


def compute_coeffs(
    density_func,
    nmax,
    lmax,
    M,
    r_s,
    args=(),
    skip_odd=False,
    skip_even=False,
    skip_m=False,
    S_only=False,
    progress=False,
    **nquad_opts
):
    """
    Compute the expansion coefficients for representing the input density
    function using a basis function expansion.

    Computing the coefficients involves computing triple integrals which are
    computationally expensive. For an example of how to parallelize the
    computation of the coefficients, see ``examples/parallel_compute_Anlm.py``.

    Parameters
    ----------
    density_func : function, callable
        A function or callable object that evaluates the density at a given
        position. The call format must be of the form: ``density_func(x, y, z,
        M, r_s, args)`` where ``x, y, z`` are cartesian coordinates, ``M`` is a
        scale mass, ``r_s`` a scale radius, and ``args`` is an iterable
        containing any other arguments needed by the density function.
    nmax : int
        Maximum value of ``n`` for the radial expansion.
    lmax : int
        Maximum value of ``l`` for the spherical harmonics.
    M : numeric
        Scale mass.
    r_s : numeric
        Scale radius.
    args : iterable (optional)
        A list or iterable of any other arguments needed by the density
        function.
    skip_odd : bool (optional)
        Skip the odd terms in the angular portion of the expansion. For example,
        only take :math:`l=0, 2, 4, ...`
    skip_even : bool (optional)
        Skip the even terms in the angular portion of the expansion. For
        example, only take :math:`l=1, 3, 5, ...`
    skip_m : bool (optional)
        Ignore terms with :math:`m > 0`.
    S_only : bool (optional)
        Only compute the S coefficients.
    progress : bool (optional)
        If ``tqdm`` is installed, display a progress bar.
    **nquad_opts
        Any additional keyword arguments are passed through to
        `~scipy.integrate.nquad` as options, `opts`.

    Returns
    -------
    Snlm : float, `~numpy.ndarray`
        The value of the cosine expansion coefficient.
    Snlm_err : , `~numpy.ndarray`
        An estimate of the uncertainty in the coefficient value (from `~scipy.integrate.nquad`).
    Tnlm : , `~numpy.ndarray`
        The value of the sine expansion coefficient.
    Tnlm_err : , `~numpy.ndarray`
        An estimate of the uncertainty in the coefficient value. (from `~scipy.integrate.nquad`).

    """
    from gala._cconfig import GSL_ENABLED

    if not GSL_ENABLED:
        raise ValueError(
            "Gala was compiled without GSL and so this function "
            "will not work.  See the gala documentation for more "
            "information about installing and using GSL with "
            "gala: http://gala.adrian.pw/en/latest/install.html"
        )

    lmin = 0
    lstride = 1

    if skip_odd or skip_even:
        lstride = 2

    if skip_even:
        lmin = 1

    Snlm = np.zeros((nmax + 1, lmax + 1, lmax + 1))
    Snlm_e = np.zeros((nmax + 1, lmax + 1, lmax + 1))
    Tnlm = np.zeros((nmax + 1, lmax + 1, lmax + 1))
    Tnlm_e = np.zeros((nmax + 1, lmax + 1, lmax + 1))

    nquad_opts.setdefault("limit", 256)
    nquad_opts.setdefault("epsrel", 1e-10)

    limits = [
        [0, 2 * np.pi],  # phi
        [-1, 1.0],  # X (cos(theta))
        [-1, 1.0],
    ]  # xsi

    nlms = []
    for n in range(nmax + 1):
        for l in range(lmin, lmax + 1, lstride):
            for m in range(l + 1):
                if skip_m and m > 0:
                    continue

                nlms.append((n, l, m))

    if progress:
        try:
            from tqdm import tqdm
        except ImportError as e:
            raise ImportError(
                "tqdm is not installed - you can install it "
                "with `pip install tqdm`.\n" + str(e)
            )
        iterfunc = tqdm
    else:
        iterfunc = lambda x: x  # noqa

    for n, l, m in iterfunc(nlms):
        Snlm[n, l, m], Snlm_e[n, l, m] = si.nquad(
            Snlm_integrand,
            ranges=limits,
            args=(density_func, n, l, m, M, r_s, args),
            opts=nquad_opts,
        )

        if not S_only:
            Tnlm[n, l, m], Tnlm_e[n, l, m] = si.nquad(
                Tnlm_integrand,
                ranges=limits,
                args=(density_func, n, l, m, M, r_s, args),
                opts=nquad_opts,
            )

    return (Snlm, Snlm_e), (Tnlm, Tnlm_e)


def compute_coeffs_discrete(
    xyz,
    mass,
    nmax,
    lmax,
    r_s,
    skip_odd=False,
    skip_even=False,
    skip_m=False,
    compute_var=False,
):
    """
    Compute the expansion coefficients for representing the density distribution
    of input points as a basis function expansion. The points, ``xyz``, are
    assumed to be samples from the density distribution.

    Computing the coefficients involves computing triple integrals which are
    computationally expensive. For an example of how to parallelize the
    computation of the coefficients, see ``examples/parallel_compute_Anlm.py``.

    Parameters
    ----------
    xyz : array_like
        Samples from the density distribution. Should have shape ``(n_samples,
        3)``.
    mass : array_like
        Mass of each sample. Should have shape ``(n_samples,)``.
    nmax : int
        Maximum value of ``n`` for the radial expansion.
    lmax : int
        Maximum value of ``l`` for the spherical harmonics.
    r_s : numeric
        Scale radius.
    skip_odd : bool (optional)
        Skip the odd terms in the angular portion of the expansion. For example,
        only take :math:`l=0, 2, 4, ...`
    skip_even : bool (optional)
        Skip the even terms in the angular portion of the expansion. For
        example, only take :math:`l=1, 3, 5, ...`
    skip_m : bool (optional)
        Ignore terms with :math:`m > 0`.
    compute_var : bool (optional)
        Also compute the variances (and covariances) of the coefficients.

    Returns
    -------
    Snlm : `~numpy.ndarray`
        The value of the cosine expansion coefficient.
    Tnlm : `~numpy.ndarray`
        The value of the sine expansion coefficient.
    STcovar : `~numpy.ndarray`, optional
        If ``compute_var==True``, this also computes and returns the covariance
        matrix of the coefficients.

    """
    lmin = 0
    lstride = 1

    if skip_odd or skip_even:
        lstride = 2

    if skip_even:
        lmin = 1

    Snlm = np.zeros((nmax + 1, lmax + 1, lmax + 1))
    Tnlm = np.zeros((nmax + 1, lmax + 1, lmax + 1))

    if compute_var:
        Snlm_var = np.zeros((nmax + 1, lmax + 1, lmax + 1))
        Tnlm_var = np.zeros((nmax + 1, lmax + 1, lmax + 1))
        STnlm_var = np.zeros((nmax + 1, lmax + 1, lmax + 1))

    # positions and masses of point masses
    xyz = np.ascontiguousarray(np.atleast_2d(xyz))
    mass = np.ascontiguousarray(np.atleast_1d(mass))

    r = np.sqrt(np.sum(xyz ** 2, axis=-1))
    s = r / r_s
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    X = xyz[:, 2] / r

    for n in range(nmax + 1):
        for l in range(lmin, lmax + 1, lstride):
            for m in range(l + 1):
                if skip_m and m > 0:
                    continue

                Snlm[n, l, m], Tnlm[n, l, m] = STnlm_discrete(
                    s, phi, X, mass, n, l, m
                )
                if compute_var:
                    (
                        Snlm_var[n, l, m],
                        Tnlm_var[n, l, m],
                        STnlm_var[n, l, m],
                    ) = STnlm_var_discrete(s, phi, X, mass, n, l, m)
    if compute_var:
        return (
            Snlm,
            Tnlm,
            np.array([[Snlm_var, STnlm_var], [STnlm_var, Tnlm_var]]),
        )
    else:
        return Snlm, Tnlm
