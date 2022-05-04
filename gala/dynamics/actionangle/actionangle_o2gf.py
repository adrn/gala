"""
Utilities for estimating actions and angles for an arbitrary orbit in an
arbitrary potential.
"""

# Standard library
import time
import warnings

# Third-party
from astropy.constants import G
import astropy.table as at
import astropy.units as u
from astropy.utils.decorators import deprecated
import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize

# Project
from gala.logging import logger
from gala.util import GalaDeprecationWarning

__all__ = ['generate_n_vectors', 'fit_isochrone',
           'fit_harmonic_oscillator', 'fit_toy_potential', 'check_angle_sampling',
           'find_actions_o2gf', 'find_actions']


def generate_n_vectors(N_max, dx=1, dy=1, dz=1, half_lattice=True):
    r"""
    Generate integer vectors, :math:`\boldsymbol{n}`, with
    :math:`|\boldsymbol{n}| < N_{\rm max}`.

    If ``half_lattice=True``, only return half of the three-dimensional
    lattice. If the set N = {(i, j, k)} defines the lattice, we restrict to
    the cases such that ``(k > 0)``, ``(k = 0, j > 0)``, and
    ``(k = 0, j = 0, i > 0)``.

    .. todo::

        Return shape should be (3, N) to be consistent.

    Parameters
    ----------
    N_max : int
        Maximum norm of the integer vector.
    dx : int
        Step size in x direction. Set to 1 for odd and even terms, set
        to 2 for just even terms.
    dy : int
        Step size in y direction. Set to 1 for odd and even terms, set
        to 2 for just even terms.
    dz : int
        Step size in z direction. Set to 1 for odd and even terms, set
        to 2 for just even terms.
    half_lattice : bool (optional)
        Only return half of the 3D lattice.

    Returns
    -------
    vecs : :class:`numpy.ndarray`
        A 2D array of integers with :math:`|\boldsymbol{n}| < N_{\rm max}`
        with shape (N, 3).

    """
    vecs = np.meshgrid(np.arange(-N_max, N_max+1, dx),
                       np.arange(-N_max, N_max+1, dy),
                       np.arange(-N_max, N_max+1, dz))
    vecs = np.vstack(list(map(np.ravel, vecs))).T
    vecs = vecs[np.linalg.norm(vecs, axis=1) <= N_max]

    if half_lattice:
        ix = ((vecs[:, 2] > 0) |
              ((vecs[:, 2] == 0) &
               (vecs[:, 1] > 0)) |
              ((vecs[:, 2] == 0) &
               (vecs[:, 1] == 0) &
               (vecs[:, 0] > 0)))
        vecs = vecs[ix]

    vecs = np.array(sorted(vecs, key=lambda x: (x[0], x[1], x[2])))
    return vecs


@u.quantity_input(m0=u.Msun, b0=u.kpc)
def fit_isochrone(orbit, m0=None, b0=None, minimize_kwargs=None):
    r"""
    Fit the toy Isochrone potential to the sum of the energy residuals relative
    to the mean energy by minimizing the function

    .. math::

        f(m, b) = \sum_i (\frac{1}{2}v_i^2 + \Phi_{\rm iso}(x_i\,|\,m, b) - <E>)^2

    TODO: This should fail if the Hamiltonian associated with the orbit has
          a frame other than StaticFrame

    Parameters
    ----------
    orbit : `~gala.dynamics.Orbit`
    m0 : numeric (optional)
        Initial guess for mass parameter of fitted Isochrone model.
    b0 : numeric (optional)
        Initial guess for scale length parameter of fitted Isochrone model.
    minimize_kwargs : dict (optional)
        Keyword arguments to pass through to `scipy.optimize.minimize`.

    Returns
    -------
    fit_iso : `gala.potential.IsochronePotential`
        Best-fit Isochrone potential for locally representing true potential.

    """
    from gala.potential import IsochronePotential, LogarithmicPotential

    pot = orbit.potential
    if pot is None:
        raise ValueError(
            "The inputted orbit does not have an associated potential instance "
            "(i.e. orbit.potential is None). You must provide an orbit instance"
            " with a specified potential in order to initialize the toy "
            "potential fitting."
        )

    w = np.squeeze(orbit.w(pot.units))
    if w.ndim > 2:
        raise ValueError("Input orbit object must be a single orbit.")

    if (m0 is not None and b0 is None) or (m0 is None and b0 is not None):
        raise ValueError("If passing in initial guess for one parameter, you "
                         "must also pass in an initial guess for the other "
                         "(m0 and b0).")

    elif m0 is not None and b0 is not None:
        # both initial guesses provided
        m0 = m0.decompose(pot.units).value
        b0 = b0.decompose(pot.units).value

    else:
        # initial guess not specified: some magic to come up with initialization
        r0 = np.mean(orbit.physicsspherical.r)
        Menc0 = pot.mass_enclosed([1, 0, 0] * r0)[0].decompose(pot.units).value
        Phi0 = pot.energy([1, 0, 0] * r0)[0]
        Phi0 = Phi0.decompose(pot.units).value
        r0 = r0.decompose(pot.units).value

        _G = G.decompose(pot.units).value

        # Special case the logarithmic potential:
        if isinstance(pot, LogarithmicPotential):
            def func(pars, r0, M0, Phi0):
                b, const = pars
                a0 = np.sqrt(r0**2 + b**2)
                return (-_G * M0 / r0**3 * a0 * (b + a0) - Phi0 + const) ** 2

            res = minimize(
                func, x0=[r0, 0],
                args=(r0, Menc0, Phi0),
                method='L-BFGS-B',
                bounds=[(0, None), (None, None)]
            )

        else:
            def func(b, r0, M0, Phi0):
                a0 = np.sqrt(r0**2 + b**2)
                return (-_G * M0 / r0**3 * a0 * (b + a0) - Phi0) ** 2

            res = minimize(
                func, x0=[r0],
                args=(r0, Menc0, Phi0),
                method='L-BFGS-B',
                bounds=[(0, None)]
            )

        if not res.success:
            raise RuntimeError(
                "Root finding failed: Unable to find local Isochrone potential "
                "fit for orbit."
            )

        b = res.x[0]
        a0 = np.sqrt(b**2 + r0**2)
        M = Menc0 / r0**3 * a0 * (b + a0)**2

        m0 = M
        b0 = b

    def f(p, w):
        logm, logb = p
        potential = IsochronePotential(m=np.exp(logm), b=np.exp(logb),
                                       units=pot.units)
        H = (potential.energy(w[:3]).decompose(pot.units).value +
             0.5*np.sum(w[3:]**2, axis=0))
        return np.sum(np.squeeze(H - np.mean(H))**2)

    logm0 = np.log(m0)
    logb0 = np.log(b0)

    if minimize_kwargs is None:
        minimize_kwargs = dict()
    minimize_kwargs.setdefault('x0', np.array([logm0, logb0]))
    minimize_kwargs.setdefault('method', 'powell')
    res = minimize(f, args=(w,), **minimize_kwargs)

    if not res.success:
        raise ValueError("Failed to fit toy potential to orbit.")

    return IsochronePotential(*np.exp(res.x), units=pot.units)


def fit_harmonic_oscillator(orbit, omega0=None, minimize_kwargs=None):
    r"""
    Fit the toy harmonic oscillator potential to the sum of the energy
    residuals relative to the mean energy by minimizing the function

    .. math::

        f(\boldsymbol{\omega}) = \sum_i (\frac{1}{2}v_i^2 +
            \Phi_{\rm sho}(x_i\,|\,\boldsymbol{\omega}) - <E>)^2

    TODO: This should fail if the Hamiltonian associated with the orbit has
          a frame other than StaticFrame

    Parameters
    ----------
    orbit : `~gala.dynamics.Orbit`
    omega0 : array_like (optional)
        Initial frequency guess.
    minimize_kwargs : dict (optional)
        Keyword arguments to pass through to `scipy.optimize.minimize`.

    Returns
    -------
    omegas : float
        Best-fit harmonic oscillator frequencies.

    """
    from gala.potential import HarmonicOscillatorPotential

    pot = orbit.potential
    if pot is None:
        raise ValueError(
            "The inputted orbit does not have an associated potential instance "
            "(i.e. orbit.potential is None). You must provide an orbit instance"
            " with a specified potential in order to initialize the toy "
            "potential fitting."
        )

    if omega0 is None:
        # Estimate from orbit:
        P = orbit.cartesian.estimate_period()[0]
        P = u.Quantity([P[k] for k in P.colnames])
        omega0 = (2*np.pi / P).decompose(pot.units).value
    else:
        omega0 = np.atleast_1d(omega0)

    w = np.squeeze(orbit.w(pot.units))
    if w.ndim > 2:
        raise ValueError("Input orbit object must be a single orbit.")

    def f(omega, w):
        potential = HarmonicOscillatorPotential(omega=omega, units=pot.units)
        H = (potential.energy(w[:3]).decompose(pot.units).value +
             0.5*np.sum(w[3:]**2, axis=0))
        return np.sum(np.squeeze(H - np.mean(H))**2)

    if minimize_kwargs is None:
        minimize_kwargs = dict()
    minimize_kwargs['x0'] = omega0
    minimize_kwargs['method'] = minimize_kwargs.get('method', 'powell')
    res = minimize(f, args=(w,), **minimize_kwargs)

    if not res.success:
        raise ValueError("Failed to fit toy potential to orbit.")

    best_omega = np.abs(res.x)
    return HarmonicOscillatorPotential(omega=best_omega, units=pot.units)


def fit_toy_potential(orbit, force_harmonic_oscillator=False, **kwargs):
    """
    Fit a best fitting toy potential to the orbit provided. If the orbit is a
    tube (loop) orbit, use the Isochrone potential. If the orbit is a box
    potential, use the harmonic oscillator potential. An option is available to
    force using the harmonic oscillator (`force_harmonic_oscillator`).

    See the docstrings for ~`gala.dynamics.fit_isochrone()` and
    ~`gala.dynamics.fit_harmonic_oscillator()` for more information.

    Parameters
    ----------
    orbit : `~gala.dynamics.Orbit`
    force_harmonic_oscillator : bool (optional)
        Force using the harmonic oscillator potential as the toy potential.

    Returns
    -------
    potential
        The best-fit potential instance.

    """

    circulation = orbit.circulation()
    if np.any(circulation == 1) and not force_harmonic_oscillator:  # tube orbit
        logger.debug("===== Tube orbit =====")
        logger.debug("Using Isochrone toy potential")

        toy_potential = fit_isochrone(orbit, **kwargs)
        logger.debug(
            f"Best m={toy_potential.parameters['m']}, "
            f"b={toy_potential.parameters['b']}"
        )

    else:  # box orbit
        logger.debug("===== Box orbit =====")
        logger.debug("Using triaxial harmonic oscillator toy potential")

        toy_potential = fit_harmonic_oscillator(orbit, **kwargs)
        logger.debug(f"Best omegas ({toy_potential.parameters['omega']})")

    return toy_potential


def check_angle_sampling(nvecs, angles):
    """
    Returns a list of the index of elements of n which do not have adequate
    toy angle coverage. The criterion is that we must have at least one sample
    in each Nyquist box when we project the toy angles along the vector n.

    Parameters
    ----------
    nvecs : array_like
        Array of integer vectors.
    angles : array_like
        Array of angles.

    Returns
    -------
    failed_nvecs : :class:`numpy.ndarray`
        Array of all integer vectors that failed checks. Has shape (N, 3).
    failures : :class:`numpy.ndarray`
        Array of flags that designate whether this failed needing a longer
        integration window (0) or finer sampling (1).

    """

    failed_nvecs = []
    failures = []
    warn_longer_window = []
    warn_finer_sampling = []
    for i, vec in enumerate(nvecs):
        # N = np.linalg.norm(vec)
        # X = np.dot(angles, vec)
        X = (angles*vec[:, None]).sum(axis=0)
        diff = float(np.abs(X.max() - X.min()))

        if diff < (2.*np.pi):
            failed_nvecs.append(vec.tolist())
            # P.append(2.*np.pi - diff)
            failures.append(0)
            warn_longer_window.append(vec)

        elif (diff/len(X)) > np.pi:
            failed_nvecs.append(vec.tolist())
            # P.append(np.pi - diff/len(X))
            failures.append(1)
            warn_finer_sampling.append(vec)

    if len(warn_longer_window) > 0:
        warn_longer_window = np.array(warn_longer_window)
        warnings.warn(
            f"Need a longer integration window for modes: {warn_longer_window}",
            RuntimeWarning
        )

    if len(warn_finer_sampling) > 0:
        warn_finer_sampling = np.array(warn_finer_sampling)
        warnings.warn(
            f"Need a finer time sampling for modes: {warn_finer_sampling}",
            RuntimeWarning
        )

    return np.array(failed_nvecs), np.array(failures)


def _action_prepare(aa, N_max, dx, dy, dz, sign=1., throw_out_modes=False):
    """
    Given toy actions and angles, `aa`, compute the matrix `A` and
    vector `b` to solve for the vector of "true" actions and generating
    function values, `x` (see Equations 12-14 in Sanders & Binney (2014)).

    .. todo::

        Wrong shape for aa -- should be (6, n) as usual...

    Parameters
    ----------
    aa : array_like
        Shape ``(6, ntimes)`` array of toy actions and angles.
    N_max : int
        Maximum norm of the integer vector.
    dx : int
        Step size in x direction. Set to 1 for odd and even terms, set
        to 2 for just even terms.
    dy : int
        Step size in y direction. Set to 1 for odd and even terms, set
        to 2 for just even terms.
    dz : int
        Step size in z direction. Set to 1 for odd and even terms, set
        to 2 for just even terms.
    sign : numeric (optional)
        Vector that defines direction of circulation about the axes.
    """

    # unroll the angles so they increase continuously instead of wrap
    angles = np.unwrap(aa[3:])

    # generate integer vectors for fourier modes
    nvecs = generate_n_vectors(N_max, dx, dy, dz)

    # make sure we have enough angle coverage
    modes, P = check_angle_sampling(nvecs, angles)

    # throw out modes?
    # if throw_out_modes:
    #     nvecs = np.delete(nvecs, (modes, P), axis=0)

    n = len(nvecs) + 3
    b = np.zeros(shape=(n, ))
    A = np.zeros(shape=(n, n))

    # top left block matrix: identity matrix summed over timesteps
    A[:3, :3] = aa.shape[1]*np.identity(3)

    actions = aa[:3]
    angles = aa[3:]

    # top right block matrix: transpose of C_nk matrix (Eq. 12)
    C_T = 2.*nvecs.T * np.sum(np.cos(np.dot(nvecs, angles)), axis=-1)
    A[:3, 3:] = C_T
    A[3:, :3] = C_T.T

    # lower right block matrix: C_nk dotted with C_nk^T
    cosv = np.cos(np.dot(nvecs, angles))
    A[3:, 3:] = 4.*np.dot(nvecs, nvecs.T)*np.einsum('it, jt->ij', cosv, cosv)

    # b vector first three is just sum of toy actions
    b[:3] = np.sum(actions, axis=1)

    # rest of the vector is C dotted with actions
    b[3:] = 2*np.sum(np.dot(nvecs, actions)*np.cos(np.dot(nvecs, angles)),
                     axis=1)

    return A, b, nvecs


def _angle_prepare(aa, t, N_max, dx, dy, dz, sign=1.):
    """
    Given toy actions and angles, `aa`, compute the matrix `A` and
    vector `b` to solve for the vector of "true" angles, frequencies, and
    generating function derivatives, `x` (see Appendix of
    Sanders & Binney (2014)).

    .. todo::

        Wrong shape for aa -- should be (6, n) as usual...

    Parameters
    ----------
    aa : array_like
        Shape ``(6, ntimes)`` array of toy actions and angles.
    t : array_like
        Array of times.
    N_max : int
        Maximum norm of the integer vector.
    dx : int
        Step size in x direction. Set to 1 for odd and even terms, set
        to 2 for just even terms.
    dy : int
        Step size in y direction. Set to 1 for odd and even terms, set
        to 2 for just even terms.
    dz : int
        Step size in z direction. Set to 1 for odd and even terms, set
        to 2 for just even terms.
    sign : numeric (optional)
        Vector that defines direction of circulation about the axes.
    """

    # unroll the angles so they increase continuously instead of wrap
    angles = np.unwrap(aa[3:])

    # generate integer vectors for fourier modes
    nvecs = generate_n_vectors(N_max, dx, dy, dz)

    # make sure we have enough angle coverage
    modes, P = check_angle_sampling(nvecs, angles)

    # TODO: throw out modes?
    # if(throw_out_modes):
    #     n_vectors = np.delete(n_vectors, check_each_direction(n_vectors, angs), axis=0)

    nv = len(nvecs)
    n = 3 + 3 + 3 * nv  # angle(0)'s, freqs, 3 derivatives of Sn

    b = np.zeros(shape=(n,))
    A = np.zeros(shape=(n, n))

    # top left block matrix: identity matrix summed over timesteps
    A[:3, :3] = aa.shape[1]*np.identity(3)

    # identity matrices summed over times
    A[:3, 3:6] = A[3:6, :3] = np.sum(t)*np.identity(3)
    A[3:6, 3:6] = np.sum(t*t)*np.identity(3)

    # S1, 2, 3
    A[6:6+nv, 0] = -2.*np.sum(np.sin(np.dot(nvecs, angles)), axis=1)
    A[6+nv:6+2*nv, 1] = A[6:6+nv, 0]
    A[6+2*nv:6+3*nv, 2] = A[6:6+nv, 0]

    # t*S1, 2, 3
    A[6:6+nv, 3] = -2.*np.sum(t[None, :]*np.sin(np.dot(nvecs, angles)),
                              axis=1)
    A[6+nv:6+2*nv, 4] = A[6:6+nv, 3]
    A[6+2*nv:6+3*nv, 5] = A[6:6+nv, 3]

    # lower right block structure: S dot S^T
    sinv = np.sin(np.dot(nvecs, angles))
    SdotST = np.einsum('it, jt->ij', sinv, sinv)
    A[6:6+nv, 6:6+nv] = A[6+nv:6+2*nv, 6+nv:6+2*nv] = \
        A[6+2*nv:6+3*nv, 6+2*nv:6+3*nv] = 4*SdotST

    # top rectangle
    A[:6, :] = A[:, :6].T

    b[:3] = np.sum(angles.T, axis=0)
    b[3:6] = np.sum(t[:, None]*angles.T, axis=0)
    b[6:6+nv] = -2.*np.sum(angles[0]*np.sin(np.dot(nvecs, angles)), axis=1)
    b[6+nv:6+2*nv] = -2.*np.sum(angles[1]*np.sin(np.dot(nvecs, angles)),
                                axis=1)
    b[6+2*nv:6+3*nv] = -2.*np.sum(angles[2]*np.sin(np.dot(nvecs, angles)),
                                  axis=1)

    return A, b, nvecs


def _single_orbit_find_actions(orbit, N_max, toy_potential=None,
                               force_harmonic_oscillator=False,
                               fit_kwargs=None):
    """
    Find approximate actions and angles for samples of a phase-space orbit,
    `w`, at times `t`. Uses toy potentials with known, analytic action-angle
    transformations to approximate the true coordinates as a Fourier sum.

    This code is adapted from Jason Sanders'
    `genfunc <https://github.com/jlsanders/genfunc>`_

    .. todo::

        Wrong shape for w -- should be (6, n) as usual...

    Parameters
    ----------
    orbit : `~gala.dynamics.Orbit`
    N_max : int
        Maximum integer Fourier mode vector length, |n|.
    toy_potential : Potential (optional)
        Fix the toy potential class.
    force_harmonic_oscillator : bool (optional)
        Force using the harmonic oscillator potential as the toy potential.
    fit_kwargs : dict (optional)
        Passed to ``fit_toy_potential()`` and on to the toy potential fitting
        functions.
    """
    from gala.potential import HarmonicOscillatorPotential, IsochronePotential

    if orbit.norbits > 1:
        raise ValueError("must be a single orbit")

    if fit_kwargs is None:
        fit_kwargs = {}

    if toy_potential is None:
        toy_potential = fit_toy_potential(
            orbit, force_harmonic_oscillator=force_harmonic_oscillator,
            **fit_kwargs)

    else:
        logger.debug(f"Using *fixed* toy potential: {toy_potential.parameters}")

    if isinstance(toy_potential, IsochronePotential):
        orbit_align = orbit.align_circulation_with_z()
        w = orbit_align.w()

        dxyz = (1, 2, 2)
        circ = np.sign(w[0, 0]*w[4, 0]-w[1, 0]*w[3, 0])
        sign = np.array([1., circ, 1.])
        orbit = orbit_align
    elif isinstance(toy_potential, HarmonicOscillatorPotential):
        dxyz = (2, 2, 2)
        sign = 1.
        w = orbit.w()
    else:
        raise ValueError("Invalid toy potential.")

    t = orbit.t.value

    # Now find toy actions and angles
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

    t1 = time.time()
    A, b, nvecs = _action_prepare(aa, N_max, dx=dxyz[0], dy=dxyz[1], dz=dxyz[2])
    actions = np.array(solve(A, b))
    logger.debug("Action solution found for N_max={}, size {} symmetric"
                 " matrix in {} seconds"
                 .format(N_max, len(actions), time.time()-t1))

    t1 = time.time()
    A, b, nvecs = _angle_prepare(aa, t, N_max, dx=dxyz[0],
                                 dy=dxyz[1], dz=dxyz[2], sign=sign)
    angles = np.array(solve(A, b))
    logger.debug("Angle solution found for N_max={}, size {} symmetric"
                 " matrix in {} seconds"
                 .format(N_max, len(angles), time.time()-t1))

    # Just some checks
    if len(angles) > len(aa):
        warnings.warn("More unknowns than equations!")

    J = actions[:3]  # * sign
    theta = angles[:3]
    freqs = angles[3:6]  # * sign

    return dict(actions=J * aaf[0].unit,
                angles=theta * aaf[1].unit,
                freqs=freqs * aaf[2].unit,
                Sn=actions[3:],
                dSn_dJ=angles[6:],
                nvecs=nvecs)


def find_actions_o2gf(orbit, N_max, force_harmonic_oscillator=False,
                      toy_potential=None, fit_kwargs=None):
    """
    Find approximate actions and angles for samples of a phase-space orbit.
    Uses toy potentials with known, analytic action-angle transformations to
    approximate the true coordinates as a Fourier sum.

    This code is adapted from Jason Sanders'
    `genfunc <https://github.com/jlsanders/genfunc>`_

    Parameters
    ----------
    orbit : `~gala.dynamics.Orbit`
    N_max : int
        Maximum integer Fourier mode vector length, :math:`|\boldsymbol{n}|`.
    force_harmonic_oscillator : bool (optional)
        Force using the harmonic oscillator potential as the toy potential.
    toy_potential : Potential (optional)
        Fix the toy potential class.

    Returns
    -------
    aaf : `astropy.table.QTable`
        An Astropy table containing the actions, angles, and frequencies for
        each input phase-space position or orbit. The columns also contain the
        value of the generating function and derivatives for each integer
        vector.

    """

    if orbit.norbits == 1:
        result = _single_orbit_find_actions(
            orbit, N_max,
            force_harmonic_oscillator=force_harmonic_oscillator,
            toy_potential=toy_potential,
            fit_kwargs=fit_kwargs
        )
        rows = [result]

    else:
        rows = []
        for n in range(orbit.norbits):
            aaf = _single_orbit_find_actions(
                orbit[:, n], N_max,
                force_harmonic_oscillator=force_harmonic_oscillator,
                toy_potential=toy_potential,
                fit_kwargs=fit_kwargs
            )

            rows.append(aaf)

    return at.QTable(rows=rows)


@deprecated(since="v1.5",
            name="find_actions",
            alternative="find_actions_o2gf",
            warning_type=GalaDeprecationWarning)
def find_actions(*args, **kwargs):
    """
    Deprecated! Use `gala.dynamics.actionangle.find_actions_o2gf` instead.
    """
    return find_actions_o2gf(*args, **kwargs)


# def solve_hessian(relative_actions, relative_freqs):
#     """ Use ordinary least squares to solve for the Hessian, given a
#         set of actions and frequencies relative to the parent orbit.
#     """

# def compute_hessian(t, w, actions_kwargs={}):
#     """ Compute the Hessian (in action-space) of the given orbit

#     """

#     N = dJ.shape[0]

#     Y = np.ravel(dF)
#     A = np.zeros((3*N, 9))
#     A[::3, :3] = dJ
#     A[1::3, 3:6] = dJ
#     A[2::3, 6:9] = dJ

#     # Solve for 'parameters' - the Hessian elements
#     X, res, rank, s = np.linalg.lstsq(A, Y)

#     # Symmetrize
#     D0 = X.reshape(3, 3)
#     D0[0, 1] = D0[1, 0] = (D0[0, 1] + D0[1, 0])/2.
#     D0[0, 2] = D0[2, 0] = (D0[0, 2] + D0[2, 0])/2.
#     D0[1, 2] = D0[2, 1] = (D0[1, 2] + D0[2, 1])/2.

#     print("Residual: " + str(res[0]))

#     return D0, np.linalg.eigh(D0) # symmetric matrix
