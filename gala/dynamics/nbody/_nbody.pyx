# # cython: boundscheck=False
# # cython: debug=False
# # cython: nonecheck=False
# # cython: cdivision=True
# # cython: wraparound=False
# # cython: profile=False
#
#
# # Standard library
# import warnings
#
# # Third-party
# import numpy as np
# cimport numpy as np
# np.import_array()
#
# from libc.math cimport sqrt
# from cpython.exc cimport PyErr_CheckSignals
#
# # from ...integrate.cyintegrators.dop853 cimport dop853_step
# # from ...potential.potential.cpotential cimport CPotentialWrapper
# # from ...potential.frame.cframe cimport CFrameWrapper
# # from ...integrate.cyintegrators.leapfrog cimport c_init_velocity, c_leapfrog_step
# # from ._coord cimport (sat_rotation_matrix, to_sat_coords, from_sat_coords,
# #                       cyl_to_car, car_to_cyl)
# #
# from ...potential import Hamiltonian
# # from ...potential.frame import StaticFrame
#
# __all__ = ['_direct_nbody_sim']
#
# # cdef extern from "frame/src/cframe.h":
# #     ctypedef struct CFrame:
# #         pass
# #
# # cdef extern from "potential/src/cpotential.h":
# #     ctypedef struct CPotential:
# #         pass
# #     double c_d2_dr2(CPotential *p, double t, double *q, double *epsilon) nogil
#
# cpdef _direct_nbody_dop853(double [:, ::1] w0, double [::1] m,
#                            external_hamiltonian,
#                            m_scale=None, r_scale=None,
#                            r_soften=None):
#     """
#     TODO
#     """
#
#     if not isinstance(external_hamiltonian, Hamiltonian):
#         raise TypeError("Input must be a Hamiltonian object, not {}"
#                         .format(type(external_hamiltonian)))
#
#     if not external_hamiltonian.c_enabled:
#         raise TypeError("Input Hamiltonian object does not support C-level access.")
#
#     cdef:
#     #     int i, j, k # indexing
#     #     int res # result from calling dop853
#     #     int ntimes = t.shape[0] # number of times
#     #     int nparticles # total number of test particles released
#     #
#     #     unsigned ndim = prog_w.shape[1] # phase-space dimensionality
#     #     unsigned ndim_2 = ndim / 2 # configuration-space dimensionality
#     #
#     #     double dt0 = t[1] - t[0] # initial timestep
#     #
#     #     double[::1] w_prime = np.zeros(6) # 6-position of stripped star
#     #     double[::1] cyl = np.zeros(6) # 6-position in cylindrical coords
#     #     double[::1] prog_w_prime = np.zeros(6) # 6-position of progenitor rotated
#     #     double[::1] prog_cyl = np.zeros(6) # 6-position of progenitor in cylindrical coords
#     #
#     #     # k-factors for parametrized model of Fardal et al. (2015)
#     #     double[::1] ks = np.zeros(6)
#     #
#     #     # used for figuring out how many orbits to integrate at any given release time
#     #     unsigned this_ndim, this_norbits
#     #
#     #     double Om # angular velocity squared
#     #     double d, sigma_r # distance, dispersion in release positions
#     #     double r_tide, menc, f # tidal radius, mass enclosed, f factor
#     #
#     #     double[::1] eps = np.zeros(3) # used for 2nd derivative estimation
#     #     double[:,::1] R = np.zeros((3,3)) # rotation matrix
#     #
#     #     double[::1] prog_mass = np.ascontiguousarray(np.atleast_1d(_prog_mass))
#     #     double[:,::1] k_mean = np.ascontiguousarray(np.atleast_2d(_k_mean))
#     #     double[:,::1] k_disp = np.ascontiguousarray(np.atleast_2d(_k_disp))
#     #     double[::1] mu_k
#     #     double[::1] sigma_k
#     #
#         CPotential cp = (<CPotentialWrapper>(hamiltonian.potential.c_instance)).cpotential
#         CFrame cf = (<CFrameWrapper>(hamiltonian.frame.c_instance)).cframe
#
#     cp.parameters
