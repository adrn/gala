# coding: utf-8
# cython: boundscheck=False
# cython: debug=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

""" Generate mock streams. """

from __future__ import division, print_function


# Standard library
import warnings

# Third-party
import numpy as np
cimport numpy as np
np.import_array()

from libc.math cimport sqrt
from cpython.exc cimport PyErr_CheckSignals

from ...integrate.cyintegrators.dop853 cimport dop853_step
from ...potential.potential.cpotential cimport CPotentialWrapper
from ...potential.frame.cframe cimport CFrameWrapper
from ...integrate.cyintegrators.leapfrog cimport c_init_velocity, c_leapfrog_step
from ._coord cimport (sat_rotation_matrix, to_sat_coords, from_sat_coords,
                      cyl_to_car, car_to_cyl)

from ...potential import Hamiltonian
from ...potential.frame import StaticFrame

__all__ = ['_mock_stream_dop853', '_mock_stream_leapfrog']

cdef extern from "frame/src/cframe.h":
    ctypedef struct CFrame:
        pass

cdef extern from "potential/src/cpotential.h":
    ctypedef struct CPotential:
        pass
    double c_d2_dr2(CPotential *p, double t, double *q, double *epsilon) nogil

cpdef _mock_stream_dop853(hamiltonian, double[::1] t, double[:,::1] prog_w,
                          int release_every,
                          _k_mean, _k_disp,
                          double G, _prog_mass, seed,
                          double atol=1E-10, double rtol=1E-10, int nmax=0):
    """
    _mock_stream_dop853(hamiltonian, t, prog_w, release_every, k_mean, k_disp, G, prog_mass, seed, atol, rtol, nmax)

    Generate a mock stellar stream using the Streakline method.

    Parameters
    ----------
    hamiltonian : `gala.potential.Hamiltonian`
        An instance of a ``Hamiltonian`` containing a gravitational potential.
    t : `numpy.ndarray`
        An array of times. Should have shape ``(ntimesteps,)``.
    prog_w : `numpy.ndarray`
        The 6D coordinates for the orbit of the progenitor system at all times.
        Should have shape ``(ntimesteps,6)``.
    release_every : int
        Release particles at the Lagrange points every X timesteps.
    k_mean : `numpy.ndarray`
        Array of mean ``k`` values (see Fardal et al. 2015). These are used to determine
        the exact prescription for generating the mock stream. The components are for:
        ``(R,phi,z,vR,vphi,vz)``. If 1D, assumed constant in time. If 2D, time axis is axis 0.
    k_disp : `numpy.ndarray`
        Array of ``k`` value dispersions (see Fardal et al. 2015). These are used to determine
        the exact prescription for generating the mock stream. The components are for:
        ``(R,phi,z,vR,vphi,vz)``. If 1D, assumed constant in time. If 2D, time axis is axis 0.
    G : numeric
        The value of the gravitational constant, G, in the unit system used.
    prog_mass : float or `numpy.ndarray`
        The mass of the progenitor or the mass at each time. Should be a scalar or have
        shape ``(ntimesteps,)``.
    seed : int (optional)
        A random number seed for initializing the particle positions.
    atol : numeric (optional)
        Passed to the integrator. Absolute tolerance parameter. Default is 1E-10.
    rtol : numeric (optional)
        Passed to the integrator. Relative tolerance parameter. Default is 1E-10.
    nmax : int (optional)
        Passed to the integrator.
    """

    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError("Input must be a Hamiltonian object, not {}".format(type(hamiltonian)))

    if not hamiltonian.c_enabled:
        raise TypeError("Input Hamiltonian object does not support C-level access.")

    cdef:
        int i, j, k # indexing
        int res # result from calling dop853
        int ntimes = t.shape[0] # number of times
        int nparticles # total number of test particles released

        unsigned ndim = prog_w.shape[1] # phase-space dimensionality
        unsigned ndim_2 = ndim / 2 # configuration-space dimensionality

        double dt0 = t[1] - t[0] # initial timestep

        double[::1] w_prime = np.zeros(6) # 6-position of stripped star
        double[::1] cyl = np.zeros(6) # 6-position in cylindrical coords
        double[::1] prog_w_prime = np.zeros(6) # 6-position of progenitor rotated
        double[::1] prog_cyl = np.zeros(6) # 6-position of progenitor in cylindrical coords

        # k-factors for parametrized model of Fardal et al. (2015)
        double[::1] ks = np.zeros(6)

        # used for figuring out how many orbits to integrate at any given release time
        unsigned this_ndim, this_norbits

        double Om # angular velocity squared
        double d, sigma_r # distance, dispersion in release positions
        double r_tide, menc, f # tidal radius, mass enclosed, f factor

        double[::1] eps = np.zeros(3) # used for 2nd derivative estimation
        double[:,::1] R = np.zeros((3,3)) # rotation matrix

        double[::1] prog_mass = np.ascontiguousarray(np.atleast_1d(_prog_mass))
        double[:,::1] k_mean = np.ascontiguousarray(np.atleast_2d(_k_mean))
        double[:,::1] k_disp = np.ascontiguousarray(np.atleast_2d(_k_disp))
        double[::1] mu_k
        double[::1] sigma_k

        # whoa, so many dots
        CPotential cp = (<CPotentialWrapper>(hamiltonian.potential.c_instance)).cpotential
        CFrame cf = (<CFrameWrapper>(hamiltonian.frame.c_instance)).cframe

    # figure out how many particles are going to be released into the "stream"
    if ntimes % release_every == 0:
        nparticles = 2 * (ntimes // release_every)
    else:
        nparticles = 2 * (ntimes // release_every + 1)

    # container for only current positions of all particles
    arr = np.empty((nparticles,ndim))
    cdef double[::1] w = arr.ravel()
    cdef double[:,::1] _w = arr

    # beginning times for each particle
    cdef double[::1] t1 = np.empty(nparticles)
    cdef double t_end = t[ntimes-1]

    # -------

    if seed is not None:
        np.random.seed(seed)

    # copy over initial conditions from progenitor orbit to each streakline star
    i = 0
    for j in range(ntimes):
        if (j % release_every) != 0:
            continue

        for k in range(ndim):
            w[2*i*ndim + k] = prog_w[j,k]
            w[2*i*ndim + k + ndim] = prog_w[j,k]

        i += 1

    # now go back to each set of initial conditions and modify initial condition
    #   based on mock prescription
    i = 0
    for j in range(ntimes):
        if (j % release_every) != 0:
            continue

        t1[2*i] = t[j]
        t1[2*i+1] = t[j]

        if prog_mass.shape[0] == 1:
            M = prog_mass[0]
        else:
            M = prog_mass[j]

        if k_mean.shape[0] == 1:
            mu_k = k_mean[0]
            sigma_k = k_disp[0]
        else:
            mu_k = k_mean[j]
            sigma_k = k_disp[j]

        # angular velocity
        d = sqrt(prog_w[j,0]*prog_w[j,0] +
                 prog_w[j,1]*prog_w[j,1] +
                 prog_w[j,2]*prog_w[j,2])
        Om = np.linalg.norm(np.cross(prog_w[j,:3], prog_w[j,3:]) / d**2)

        # gradient of potential in radial direction
        f = Om*Om - c_d2_dr2(&cp, t[j], &prog_w[j,0], &eps[0]) # TODO: do I need to use hamiltonian here?
        r_tide = (G*M / f)**(1/3.)

        # the rotation matrix to transform from satellite coords to normal
        sat_rotation_matrix(&prog_w[j,0], &R[0,0])
        to_sat_coords(&prog_w[j,0], &R[0,0], &prog_w_prime[0])
        car_to_cyl(&prog_w_prime[0], &prog_cyl[0])

        for k in range(6):
            if sigma_k[k] > 0:
                ks[k] = np.random.normal(mu_k[k], sigma_k[k])
            else:
                ks[k] = mu_k[k]

        # eject stars at tidal radius with same angular velocity as progenitor
        cyl[0] = prog_cyl[0] + ks[0]*r_tide
        cyl[1] = prog_cyl[1] + ks[1]*r_tide/prog_cyl[0]
        cyl[2] = ks[2]*r_tide/prog_cyl[0]
        cyl[3] = prog_cyl[3] + ks[3]*prog_cyl[3]
        cyl[4] = prog_cyl[4] + ks[0]*ks[4]*Om*r_tide
        cyl[5] = ks[5]*Om*r_tide
        cyl_to_car(&cyl[0], &w_prime[0])
        from_sat_coords(&w_prime[0], &R[0,0], &w[2*i*ndim])

        for k in range(6):
            if sigma_k[k] > 0:
                ks[k] = np.random.normal(mu_k[k], sigma_k[k])
            else:
                ks[k] = mu_k[k]

        cyl[0] = prog_cyl[0] - ks[0]*r_tide
        cyl[1] = prog_cyl[1] - ks[1]*r_tide/prog_cyl[0]
        cyl[2] = ks[2]*r_tide/prog_cyl[0]
        cyl[3] = prog_cyl[3] + ks[3]*prog_cyl[3]
        cyl[4] = prog_cyl[4] - ks[0]*ks[4]*Om*r_tide
        cyl[5] = ks[5]*Om*r_tide
        cyl_to_car(&cyl[0], &w_prime[0])
        from_sat_coords(&w_prime[0], &R[0,0], &w[2*i*ndim + ndim])

        i += 1

    for i in range(nparticles):
        dop853_step(&cp, &cf, &_w[i,0], t1[i], t_end, dt0,
                    ndim, 1,
                    atol, rtol, nmax)

        PyErr_CheckSignals()

    return np.asarray(w).reshape(nparticles, ndim)

cpdef _mock_stream_leapfrog(hamiltonian, double[::1] t, double[:,::1] prog_w,
                            int release_every,
                            _k_mean, _k_disp,
                            double G, double _prog_mass, seed,
                            dt=None):
    """
    _mock_stream_leapfrog(cpotential, t, prog_w, release_every, k_mean, k_disp, G, prog_mass, seed)

    Generate a mock stellar stream using the Streakline method.

    Parameters
    ----------
    hamiltonian : `gala.potential.Hamiltonian`
        An instance of a ``Hamiltonian`` containing a gravitational potential.
    t : `numpy.ndarray`
        An array of times. Should have shape ``(ntimesteps,)``.
    prog_w : `numpy.ndarray`
        The 6D coordinates for the orbit of the progenitor system at all times.
        Should have shape ``(ntimesteps,6)``.
    release_every : int
        Release particles at the Lagrange points every X timesteps.
    k_mean : `numpy.ndarray`
        Array of mean ``k`` values (see Fardal et al. 2015). These are used to determine
        the exact prescription for generating the mock stream. The components are for:
        ``(R,phi,z,vR,vphi,vz)``. If 1D, assumed constant in time. If 2D, time axis is axis 0.
    k_disp : `numpy.ndarray`
        Array of ``k`` value dispersions (see Fardal et al. 2015). These are used to determine
        the exact prescription for generating the mock stream. The components are for:
        ``(R,phi,z,vR,vphi,vz)``. If 1D, assumed constant in time. If 2D, time axis is axis 0.
    G : numeric
        The value of the gravitational constant, G, in the unit system used.
    prog_mass : float or `numpy.ndarray`
        The mass of the progenitor or the mass at each time. Should be a scalar or have
        shape ``(ntimesteps,)``.
    seed : int (optional)
        A random number seed for initializing the particle positions.
    dt : float (optional)
        Timestep for integrating the orbits of each particle in the stream. Uses the
        same timestep as the progenitor orbit by default.
    """

    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError("Input must be a Hamiltonian object, not {}".format(type(hamiltonian)))

    if not hamiltonian.c_enabled:
        raise TypeError("Input Hamiltonian object does not support C-level access.")

    if not isinstance(hamiltonian.frame, StaticFrame):
        raise TypeError("Leapfrog integration is currently only supported "
                        "for StaticFrame, not {}."
                        .format(hamiltonian.frame.__class__.__name__))

    cdef:
        int i, j, k # indexing
        int ntimes = t.shape[0] # number of times
        int nparticles # total number of test particles released

        unsigned ndim = prog_w.shape[1] # phase-space dimensionality
        unsigned ndim_2 = ndim / 2 # configuration-space dimensionality

        double dt0 = t[1] - t[0] # timestep of progenitor orbit
        double _dt

        double[::1] w_prime = np.zeros(6) # 6-position of stripped star
        double[::1] cyl = np.zeros(6) # 6-position in cylindrical coords
        double[::1] prog_w_prime = np.zeros(6) # 6-position of progenitor rotated
        double[::1] prog_cyl = np.zeros(6) # 6-position of progenitor in cylindrical coords

        # k-factors for parametrized model of Fardal et al. (2015)
        double[::1] ks = np.zeros(6)

        # used for figuring out how many orbits to integrate at any given release time
        unsigned this_ndim, this_norbits

        double Om # angular velocity squared
        double d, sigma_r # distance, dispersion in release positions
        double r_tide, menc, f # tidal radius, mass enclosed, f factor

        double[::1] eps = np.zeros(3) # used for 2nd derivative estimation
        double[:,::1] R = np.zeros((3,3)) # rotation matrix

        double[::1] prog_mass = np.ascontiguousarray(np.atleast_1d(_prog_mass))
        double[:,::1] k_mean = np.ascontiguousarray(np.atleast_2d(_k_mean))
        double[:,::1] k_disp = np.ascontiguousarray(np.atleast_2d(_k_disp))
        double[::1] mu_k
        double[::1] sigma_k

        double[::1] v_jm1_2 = np.zeros(ndim_2) # temporary array to hold velocities at half step
        double[::1] grad = np.zeros(ndim_2) # temporary array to hold gradient

        CPotential cp = (<CPotentialWrapper>(hamiltonian.potential.c_instance)).cpotential

    if dt is None:
        _dt = dt0
    else:
        # TODO: implement support for non-default timestep for orbits
        # need to change the way integration is done below because of time array
        raise NotImplementedError()
        # _dt = float(dt)

    # figure out how many particles are going to be released into the "stream"
    if ntimes % release_every == 0:
        nparticles = 2 * (ntimes // release_every)
    else:
        nparticles = 2 * (ntimes // release_every + 1)

    # container for only current positions of all particles
    cdef double[::1] w = np.empty(nparticles*ndim)

    # beginning times for each particle
    cdef double[::1] t1 = np.empty(nparticles)
    cdef double t_end = t[ntimes-1]

    # -------

    if seed is not None:
        np.random.seed(seed)

    # copy over initial conditions from progenitor orbit to each streakline star
    i = 0
    for j in range(ntimes):
        if (j % release_every) != 0:
            continue

        for k in range(ndim):
            w[2*i*ndim + k] = prog_w[j,k]
            w[2*i*ndim + k + ndim] = prog_w[j,k]

        i += 1

    # now go back to each set of initial conditions and modify initial condition
    #   based on mock prescription
    i = 0
    for j in range(ntimes):
        if (j % release_every) != 0:
            continue

        t1[2*i] = t[j]
        t1[2*i+1] = t[j]

        if prog_mass.shape[0] == 1:
            M = prog_mass[0]
        else:
            M = prog_mass[j]

        if k_mean.shape[0] == 1:
            mu_k = k_mean[0]
            sigma_k = k_disp[0]
        else:
            mu_k = k_mean[j]
            sigma_k = k_disp[j]

        # angular velocity
        d = sqrt(prog_w[j,0]*prog_w[j,0] +
                 prog_w[j,1]*prog_w[j,1] +
                 prog_w[j,2]*prog_w[j,2])
        Om = np.linalg.norm(np.cross(prog_w[j,:3], prog_w[j,3:]) / d**2)

        # gradient of potential in radial direction
        f = Om*Om - c_d2_dr2(&cp, t[j], &prog_w[j,0], &eps[0])
        r_tide = (G*M / f)**(1/3.)

        # the rotation matrix to transform from satellite coords to normal
        sat_rotation_matrix(&prog_w[j,0], &R[0,0])
        to_sat_coords(&prog_w[j,0], &R[0,0], &prog_w_prime[0])
        car_to_cyl(&prog_w_prime[0], &prog_cyl[0])

        for k in range(6):
            if sigma_k[k] > 0:
                ks[k] = np.random.normal(mu_k[k], sigma_k[k])
            else:
                ks[k] = mu_k[k]

        # eject stars at tidal radius with same angular velocity as progenitor
        cyl[0] = prog_cyl[0] + ks[0]*r_tide
        cyl[1] = prog_cyl[1] + ks[1]*r_tide/prog_cyl[0]
        cyl[2] = ks[2]*r_tide/prog_cyl[0]
        cyl[3] = prog_cyl[3] + ks[3]*prog_cyl[3]
        cyl[4] = prog_cyl[4] + ks[0]*ks[4]*Om*r_tide
        cyl[5] = ks[5]*Om*r_tide
        cyl_to_car(&cyl[0], &w_prime[0])
        from_sat_coords(&w_prime[0], &R[0,0], &w[2*i*ndim])

        for k in range(6):
            if sigma_k[k] > 0:
                ks[k] = np.random.normal(mu_k[k], sigma_k[k])
            else:
                ks[k] = mu_k[k]

        cyl[0] = prog_cyl[0] - ks[0]*r_tide
        cyl[1] = prog_cyl[1] - ks[1]*r_tide/prog_cyl[0]
        cyl[2] = ks[2]*r_tide/prog_cyl[0]
        cyl[3] = prog_cyl[3] + ks[3]*prog_cyl[3]
        cyl[4] = prog_cyl[4] - ks[0]*ks[4]*Om*r_tide
        cyl[5] = ks[5]*Om*r_tide
        cyl_to_car(&cyl[0], &w_prime[0])
        from_sat_coords(&w_prime[0], &R[0,0], &w[2*i*ndim + ndim])

        i += 1

    for i in range(nparticles):
        c_init_velocity(&cp, ndim_2, t1[0], _dt,
                        &w[i*ndim], &w[i*ndim+ndim_2], &v_jm1_2[0], &grad[0])

        ntimes = int((t_end - t1[i]) / _dt)
        for j in range(ntimes):
            for k in range(ndim_2):
                grad[k] = 0.

            c_leapfrog_step(&cp, ndim_2, t1[j], _dt,
                            &w[i*ndim], &w[i*ndim+ndim_2], &v_jm1_2[0], &grad[0])

        PyErr_CheckSignals()

    return np.asarray(w).reshape(nparticles, ndim)

cpdef _mock_stream_animate(snapshot_filename, hamiltonian,
                           double[::1] t, double[:,::1] prog_w,
                           int release_every,
                           _k_mean, _k_disp,
                           double G, _prog_mass, seed,
                           double atol=1E-10, double rtol=1E-10, int nmax=0,
                           int check_filesize=1):
    """
    _mock_stream_animate(filename, cpotential, t, prog_w, release_every, k_mean, k_disp, G, prog_mass, seed, atol, rtol, nmax)

    WARNING: only use this if you want to make an animation of a stream forming!

    Parameters
    ----------
    snapshot_filename : str
        The filename of the HDF5 snapshot file to write.
    cpotential : `gala.potential._CPotential`
        An instance of a ``_CPotential`` representing the gravitational potential.
    t : `numpy.ndarray`
        An array of times. Should have shape ``(ntimesteps,)``.
    prog_w : `numpy.ndarray`
        The 6D coordinates for the orbit of the progenitor system at all times.
        Should have shape ``(ntimesteps,6)``.
    release_every : int
        Release particles at the Lagrange points every X timesteps.
    k_mean : `numpy.ndarray`
        Array of mean ``k`` values (see Fardal et al. 2015). These are used to determine
        the exact prescription for generating the mock stream. The components are for:
        ``(R,phi,z,vR,vphi,vz)``. If 1D, assumed constant in time. If 2D, time axis is axis 0.
    k_disp : `numpy.ndarray`
        Array of ``k`` value dispersions (see Fardal et al. 2015). These are used to determine
        the exact prescription for generating the mock stream. The components are for:
        ``(R,phi,z,vR,vphi,vz)``. If 1D, assumed constant in time. If 2D, time axis is axis 0.
    G : numeric
        The value of the gravitational constant, G, in the unit system used.
    prog_mass : float or `numpy.ndarray`
        The mass of the progenitor or the mass at each time. Should be a scalar or have
        shape ``(ntimesteps,)``.
    seed : int (optional)
        A random number seed for initializing the particle positions.
    atol : numeric (optional)
        Passed to the integrator. Absolute tolerance parameter. Default is 1E-10.
    rtol : numeric (optional)
        Passed to the integrator. Relative tolerance parameter. Default is 1E-10.
    nmax : int (optional)
        Passed to the integrator.
    check_filesize : bool (optional)
        Check the output filesize and warn the user if it is larger than
        10GB. Default = True.
    """

    import h5py

    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError("Input must be a Hamiltonian object, not {}".format(type(hamiltonian)))

    if not hamiltonian.c_enabled:
        raise TypeError("Input Hamiltonian object does not support C-level access.")

    cdef:
        int i, j, k # indexing
        int res # result from calling dop853
        int ntimes = t.shape[0] # number of times
        int nparticles # total number of test particles released

        unsigned ndim = prog_w.shape[1] # phase-space dimensionality
        unsigned ndim_2 = ndim / 2 # configuration-space dimensionality

        double dt0 = t[1] - t[0] # initial timestep

        double[::1] w_prime = np.zeros(6) # 6-position of stripped star
        double[::1] cyl = np.zeros(6) # 6-position in cylindrical coords
        double[::1] prog_w_prime = np.zeros(6) # 6-position of progenitor rotated
        double[::1] prog_cyl = np.zeros(6) # 6-position of progenitor in cylindrical coords

        # k-factors for parametrized model of Fardal et al. (2015)
        double[::1] ks = np.zeros(6)

        # used for figuring out how many orbits to integrate at any given release time
        unsigned this_ndim, this_norbits

        double Om # angular velocity squared
        double d, sigma_r # distance, dispersion in release positions
        double r_tide, menc, f # tidal radius, mass enclosed, f factor

        double[::1] eps = np.zeros(3) # used for 2nd derivative estimation
        double[:,::1] R = np.zeros((3,3)) # rotation matrix

        double[::1] prog_mass = np.ascontiguousarray(np.atleast_1d(_prog_mass))
        double[:,::1] k_mean = np.ascontiguousarray(np.atleast_2d(_k_mean))
        double[:,::1] k_disp = np.ascontiguousarray(np.atleast_2d(_k_disp))
        double[::1] mu_k
        double[::1] sigma_k

        double t_j

        # whoa, so many dots
        CPotential cp = (<CPotentialWrapper>(hamiltonian.potential.c_instance)).cpotential
        CFrame cf = (<CFrameWrapper>(hamiltonian.frame.c_instance)).cframe

    # figure out how many particles are going to be released into the "stream"
    if ntimes % release_every == 0:
        nparticles = 2 * (ntimes // release_every)
    else:
        nparticles = 2 * (ntimes // release_every + 1)

    # estimate size of output file and warn user if it's large
    if check_filesize:
        est_filesize_GB = nparticles * ntimes / 2 * 8 / 1024 / 1024 / 1024
        if est_filesize_GB >= 10.:
            warnings.warn("gala.dynamics.mockstream: Estimated output "
                          "filesize is >= 10 GB!")

    # container for only current positions of all particles
    cdef double[::1] w = np.empty(nparticles*ndim)
    cdef double[:,::1] one_particle_w = np.empty((ntimes,ndim))

    # beginning times for each particle
    cdef double[::1] t1 = np.empty(nparticles)
    cdef int[::1] all_ntimes = np.zeros(nparticles, dtype=np.int32)
    cdef double t_end = t[ntimes-1]

    # -------

    if seed is not None:
        np.random.seed(seed)

    # copy over initial conditions from progenitor orbit to each streakline star
    i = 0
    for j in range(ntimes):
        if (j % release_every) != 0:
            continue

        for k in range(ndim):
            w[2*i*ndim + k] = prog_w[j,k]
            w[2*i*ndim + k + ndim] = prog_w[j,k]

        i += 1

    # now go back to each set of initial conditions and modify initial condition
    #   based on mock prescription
    i = 0
    for j in range(ntimes):
        if (j % release_every) != 0:
            continue

        t1[2*i] = t[j]
        t1[2*i+1] = t[j]
        all_ntimes[2*i] = ntimes - j
        all_ntimes[2*i+1] = ntimes - j

        if prog_mass.shape[0] == 1:
            M = prog_mass[0]
        else:
            M = prog_mass[j]

        if k_mean.shape[0] == 1:
            mu_k = k_mean[0]
            sigma_k = k_disp[0]
        else:
            mu_k = k_mean[j]
            sigma_k = k_disp[j]

        # angular velocity
        d = sqrt(prog_w[j,0]*prog_w[j,0] +
                 prog_w[j,1]*prog_w[j,1] +
                 prog_w[j,2]*prog_w[j,2])
        Om = np.linalg.norm(np.cross(prog_w[j,:3], prog_w[j,3:]) / d**2)

        # gradient of potential in radial direction
        f = Om*Om - c_d2_dr2(&cp, t[j], &prog_w[j,0], &eps[0])
        r_tide = (G*M / f)**(1/3.)

        # the rotation matrix to transform from satellite coords to normal
        sat_rotation_matrix(&prog_w[j,0], &R[0,0])
        to_sat_coords(&prog_w[j,0], &R[0,0], &prog_w_prime[0])
        car_to_cyl(&prog_w_prime[0], &prog_cyl[0])

        for k in range(6):
            if sigma_k[k] > 0:
                ks[k] = np.random.normal(mu_k[k], sigma_k[k])
            else:
                ks[k] = mu_k[k]

        # eject stars at tidal radius with same angular velocity as progenitor
        cyl[0] = prog_cyl[0] + ks[0]*r_tide
        cyl[1] = prog_cyl[1] + ks[1]*r_tide/prog_cyl[0]
        cyl[2] = ks[2]*r_tide/prog_cyl[0]
        cyl[3] = prog_cyl[3] + ks[3]*prog_cyl[3]
        cyl[4] = prog_cyl[4] + ks[0]*ks[4]*Om*r_tide
        cyl[5] = ks[5]*Om*r_tide
        cyl_to_car(&cyl[0], &w_prime[0])
        from_sat_coords(&w_prime[0], &R[0,0], &w[2*i*ndim])

        for k in range(6):
            if sigma_k[k] > 0:
                ks[k] = np.random.normal(mu_k[k], sigma_k[k])
            else:
                ks[k] = mu_k[k]

        cyl[0] = prog_cyl[0] - ks[0]*r_tide
        cyl[1] = prog_cyl[1] - ks[1]*r_tide/prog_cyl[0]
        cyl[2] = ks[2]*r_tide/prog_cyl[0]
        cyl[3] = prog_cyl[3] + ks[3]*prog_cyl[3]
        cyl[4] = prog_cyl[4] - ks[0]*ks[4]*Om*r_tide
        cyl[5] = ks[5]*Om*r_tide
        cyl_to_car(&cyl[0], &w_prime[0])
        from_sat_coords(&w_prime[0], &R[0,0], &w[2*i*ndim + ndim])

        i += 1

    # create the output file
    with h5py.File(str(snapshot_filename), 'w') as h5f:
        h5f.create_dataset('pos', dtype='f8', shape=(ndim_2, ntimes, nparticles),
                           fillvalue=np.nan, compression='gzip', compression_opts=9)
        h5f.create_dataset('vel', dtype='f8', shape=(ndim_2, ntimes, nparticles),
                           fillvalue=np.nan, compression='gzip', compression_opts=9)
        h5f.create_dataset('t', data=np.array(t))

    for i in range(nparticles):
        t_j = t1[i]

        for k in range(ndim):
            one_particle_w[0,k] = w[i*ndim + k]

        for j in range(1,all_ntimes[i]+1,1):
            dop853_step(&cp, &cf, &w[i*ndim], t_j, t_j+dt0, dt0,
                        ndim, 1,
                        atol, rtol, nmax)

            PyErr_CheckSignals()

            for k in range(ndim):
                one_particle_w[j,k] = w[i*ndim + k]

            t_j = t_j+dt0

        with h5py.File(str(snapshot_filename), 'a') as h5f:
            j = ntimes - all_ntimes[i]
            h5f['pos'][:,j:,i] = np.array(one_particle_w[:all_ntimes[i], :ndim_2]).T
            h5f['vel'][:,j:,i] = np.array(one_particle_w[:all_ntimes[i], ndim_2:]).T

