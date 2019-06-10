# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3

# Standard library

# Third-party
import astropy.units as u
import numpy as np
cimport numpy as np

# This package
from .. import combine
from ..nbody import DirectNBody
from ...potential import Hamiltonian, PotentialBase

from ...potential.potential.cpotential cimport CPotentialWrapper
from ...potential.frame.cframe cimport CFrameWrapper
from ...potential.hamiltonian.chamiltonian import Hamiltonian

from ._coord cimport cross, norm, apply_3matrix

cdef extern from "potential/src/cpotential.h":
    ctypedef struct CPotential:
        pass

    # double c_potential(CPotential *p, double t, double *q) nogil
    # double c_density(CPotential *p, double t, double *q) nogil
    # void c_gradient(CPotential *p, double t, double *q, double *grad) nogil
    # void c_hessian(CPotential *p, double t, double *q, double *hess) nogil

    # double c_d_dr(CPotential *p, double t, double *q, double *epsilon) nogil
    double c_d2_dr2(CPotential *p, double t, double *q, double *epsilon) nogil
    # double c_mass_enclosed(CPotential *p, double t, double *q, double G, double *epsilon) nogil


# ---

cdef class BaseStreamDF:

    def __init__(self, H, lead=True, trail=True, **kwargs):
        self._lead = int(lead)
        self._trail = int(trail)
        self._potential = H.potential.c_instance
        self._frame = H.frame.c_instance
        self.extra_kwargs = kwargs

        # self.H = H
        self._G = H.potential.G

        if not self.lead and not self.trail:
            raise ValueError("You must generate either leading or trailing "
                             "tails (or both!)")

    cdef void get_rj_vj_R(self, double *prog_x, double *prog_v,
                          double prog_m, double t,
                          double *rj, double *vj, double[:, ::1] R): # outputs
        # HACK: assuming ndim=3 throughout here
        cdef:
            int i
            double dist = norm(prog_x, 3)
            double L[3]
            double Lmag, Om, d2r

        # angular momentum vector, L, and |L|
        cross(prog_x, prog_v, &L[0])
        Lnorm = norm(&L[0], 3)

        # TODO: note that R goes from non-rotating frame to rotating frame!!!
        for i in range(3):
            R[0, i] = prog_x[i] / dist
            R[2, i] = L[i] / Lnorm

        # Now compute jacobi radius and relative velocity at jacobi radius
        # Note: we re-use the L array as the "epsilon" array needed by d2_dr2
        Om = Lnorm / dist**2
        d2r = c_d2_dr2(&(self._potential.cpotential), t, prog_x,
                       &L[0])
        rj[0] = (self._G * prog_m / (Om*Om - d2r)) ** (1/3.)
        vj[0] = Om * rj[0]

        # re-use the epsilon array to compute cross-product
        cross(&R[0, 0], &R[2, 0], &R[1, 0])
        for i in range(3):
            R[1, i] = -R[1, i]

    cdef void transform_from_sat(self, double[:, ::1] R,
                                 double *x, double *v,
                                 double *prog_x, double *prog_v,
                                 double *out_x, double *out_v):
        # from satellite coordinates to global coordinates note: the 1 is
        # because above in get_rj_vj_R(), we compute the transpose of the
        # rotation matrix we actually need
        apply_3matrix(R, x, out_x, 1)
        apply_3matrix(R, v, out_v, 1)

        for n in range(3):
            out_x[n] += prog_x[n]
            out_v[n] += prog_v[n]


    cpdef _sample(self, double[:, ::1] prog_x, double[:, ::1] prog_v,
                  double[::1] prog_t, double[::1] prog_m, int[::1] nparticles,
                  dict extra_kwargs):
        pass

    # ------------------------------------------------------------------------
    # Python-only:

    @property
    def lead(self):
        return self._lead

    @property
    def trail(self):
        return self._trail


cdef class StreaklineStreamDF(BaseStreamDF):

    cpdef _sample(self, double[:, ::1] prog_x, double[:, ::1] prog_v,
                  double[::1] prog_t, double[::1] prog_m, int[::1] nparticles,
                  dict extra_kwargs):
        cdef:
            int i, j, k, n
            int ntimes = len(prog_t)
            int total_nparticles = (self._lead + self._trail) * np.sum(nparticles)

            double[:, ::1] particle_x = np.zeros((total_nparticles, 3))
            double[:, ::1] particle_v = np.zeros((total_nparticles, 3))
            double[::1] particle_t1 = np.zeros((total_nparticles, ))

            double[::1] tmp_x = np.zeros(3)
            double[::1] tmp_v = np.zeros(3)

            double rj # jacobi radius
            double vj # relative velocity at jacobi radius
            double[:, ::1] R = np.zeros((3, 3)) # rotation to satellite coordinates

        j = 0
        for i in range(ntimes):
            self.get_rj_vj_R(&prog_x[i, 0], &prog_v[i, 0], prog_m[i], prog_t[i],
                             &rj, &vj, R) # outputs

            # Trailing tail
            if self._trail == 1:
                for k in range(nparticles[i]):
                    tmp_x[0] = rj
                    tmp_v[1] = vj
                    particle_t1[j+k] = prog_t[i]

                    self.transform_from_sat(R,
                                            &tmp_x[0], &tmp_v[0],
                                            &prog_x[i, 0], &prog_v[i, 0],
                                            &particle_x[j+k, 0],
                                            &particle_v[j+k, 0])

                j += nparticles[i]

            # Leading tail
            if self._lead == 1:
                for k in range(nparticles[i]):
                    tmp_x[0] = -rj
                    tmp_v[1] = -vj
                    particle_t1[j+k] = prog_t[i]

                    self.transform_from_sat(R,
                                            &tmp_x[0], &tmp_v[0],
                                            &prog_x[i, 0], &prog_v[i, 0],
                                            &particle_x[j+k, 0],
                                            &particle_v[j+k, 0])

                j += nparticles[i]

        return particle_x, particle_v, particle_t1


cdef class FardalStreamDF(BaseStreamDF):

    cpdef _sample(self, double[:, ::1] prog_x, double[:, ::1] prog_v,
                  double[::1] prog_t, double[::1] prog_m, int[::1] nparticles,
                  dict extra_kwargs):
        cdef:
            int i, j, k, n
            int ntimes = len(prog_t)
            int total_nparticles = (self._lead + self._trail) * np.sum(nparticles)

            double[:, ::1] particle_x = np.zeros((total_nparticles, 3))
            double[:, ::1] particle_v = np.zeros((total_nparticles, 3))
            double[::1] particle_t1 = np.zeros((total_nparticles, ))

            double[::1] tmp_x = np.zeros(3)
            double[::1] tmp_v = np.zeros(3)

            double rj # jacobi radius
            double vj # relative velocity at jacobi radius
            double[:, ::1] R = np.zeros((3, 3)) # rotation to satellite coordinates

            # for Fardal method:
            double kx
            double[::1] k_mean = np.zeros(6)
            double[::1] k_disp = np.zeros(6)

        k_mean[0] = 2. # R
        k_disp[0] = 0.5

        k_mean[2] = 0. # z
        k_disp[2] = 0.5

        k_mean[4] = 0.3 # vt
        k_disp[4] = 0.5

        k_mean[5] = 0. # vz
        k_disp[5] = 0.5

        j = 0
        for i in range(ntimes):
            self.get_rj_vj_R(&prog_x[i, 0], &prog_v[i, 0], prog_m[i], prog_t[i],
                             &rj, &vj, R) # outputs

            # Trailing tail
            if self._trail == 1:
                for k in range(nparticles[i]):
                    kx = np.random.normal(k_mean[0], k_disp[0])
                    tmp_x[0] = kx * rj
                    tmp_x[2] = np.random.normal(k_mean[2], k_disp[2]) * rj
                    tmp_v[1] = kx * np.random.normal(k_mean[4], k_disp[4]) * vj
                    tmp_v[2] = np.random.normal(k_mean[5], k_disp[5]) * vj
                    particle_t1[j+k] = prog_t[i]

                    self.transform_from_sat(R,
                                            &tmp_x[0], &tmp_v[0],
                                            &prog_x[i, 0], &prog_v[i, 0],
                                            &particle_x[j+k, 0],
                                            &particle_v[j+k, 0])

                j += nparticles[i]

            # Leading tail
            if self._lead == 1:
                for k in range(nparticles[i]):
                    kx = np.random.normal(k_mean[0], k_disp[0])
                    tmp_x[0] = kx * -rj
                    tmp_x[2] = np.random.normal(k_mean[2], k_disp[2]) * -rj
                    tmp_v[1] = kx * np.random.normal(k_mean[4], k_disp[4]) * -vj
                    tmp_v[2] = np.random.normal(k_mean[5], k_disp[5]) * -vj
                    particle_t1[j+k] = prog_t[i]

                    self.transform_from_sat(R,
                                            &tmp_x[0], &tmp_v[0],
                                            &prog_x[i, 0], &prog_v[i, 0],
                                            &particle_x[j+k, 0],
                                            &particle_v[j+k, 0])

                j += nparticles[i]

        return particle_x, particle_v, particle_t1


cdef class MLCSStreamDF(BaseStreamDF):

    def __init__(self, H, lead=True, trail=True, ):
        super().__init__(H, lead=lead, trail=trail, extra_kwargs=dict())

    cpdef _sample(self, double[:, ::1] prog_x, double[:, ::1] prog_v,
                  double[::1] prog_t, double[::1] prog_m, int[::1] nparticles,
                  dict extra_kwargs):
        cdef:
            int i, j, k, n
            int ntimes = len(prog_t)
            int total_nparticles = (self._lead + self._trail) * np.sum(nparticles)

            double[:, ::1] particle_x = np.zeros((total_nparticles, 3))
            double[:, ::1] particle_v = np.zeros((total_nparticles, 3))
            double[::1] particle_t1 = np.zeros((total_nparticles, ))

            double[::1] tmp_x = np.zeros(3)
            double[::1] tmp_v = np.zeros(3)

            double rj # jacobi radius
            double vj # relative velocity at jacobi radius
            double[:, ::1] R = np.zeros((3, 3)) # rotation to satellite coordinates

            double[::1] v_disp = np.zeros(ntimes)

        if 'v_disp' not in extra_kwargs:
            raise ValueError('TODO: must supply a velocity dispersion...')

        _v_disp = np.array(extra_kwargs['v_disp'])
        if _v_disp.shape:
            v_disp = _v_disp
        else:
            for i in range(ntimes):
                v_disp[i] = _v_disp

        j = 0
        for i in range(ntimes):
            self.get_rj_vj_R(&prog_x[i, 0], &prog_v[i, 0], prog_m[i], prog_t[i],
                             &rj, &vj, R) # outputs

            # Trailing tail
            if self._trail == 1:
                for k in range(nparticles[i]):
                    tmp_x[0] = rj
                    tmp_v[0] = np.random.normal(0, v_disp[i])
                    tmp_v[1] = np.random.normal(0, v_disp[i])
                    tmp_v[2] = np.random.normal(0, v_disp[i])
                    particle_t1[j+k] = prog_t[i]

                    self.transform_from_sat(R,
                                            &tmp_x[0], &tmp_v[0],
                                            &prog_x[i, 0], &prog_v[i, 0],
                                            &particle_x[j+k, 0],
                                            &particle_v[j+k, 0])

                j += nparticles[i]

            # Leading tail
            if self._lead == 1:
                for k in range(nparticles[i]):
                    tmp_x[0] = -rj
                    tmp_v[0] = np.random.normal(0, v_disp[i])
                    tmp_v[1] = np.random.normal(0, v_disp[i])
                    tmp_v[2] = np.random.normal(0, v_disp[i])
                    particle_t1[j+k] = prog_t[i]

                    self.transform_from_sat(R,
                                            &tmp_x[0], &tmp_v[0],
                                            &prog_x[i, 0], &prog_v[i, 0],
                                            &particle_x[j+k, 0],
                                            &particle_v[j+k, 0])

                j += nparticles[i]

        return particle_x, particle_v, particle_t1
