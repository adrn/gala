# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3

# Third-party
from astropy.utils.misc import isiterable
import cython
import astropy.units as u
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, sin, cos, M_PI

# This package
from .. import combine, Orbit
from ..nbody import DirectNBody
from ...potential import Hamiltonian, PotentialBase, StaticFrame
from ...potential.potential.cpotential cimport CPotentialWrapper, CPotential
from ...potential.hamiltonian.chamiltonian import Hamiltonian

from ._coord cimport cross, norm, apply_3matrix
from .core import MockStream

__all__ = ['BaseStreamDF', 'FardalStreamDF', 'StreaklineStreamDF',
           'LagrangeCloudStreamDF', 'ChenStreamDF']

cdef extern from "potential/src/cpotential.h":
    double c_d2_dr2(CPotential *p, double t, double *q, double *epsilon) nogil


@cython.embedsignature(True)
cdef class BaseStreamDF:
    """A base class for representing distribution functions for generating
    stellar streams.

    This class specifies how massless star particles should be sampled in
    order to generate a mock stellar stream.

    Parameters
    ----------
    lead : bool (optional)
        Generate a leading tail. Default: True.
    trail : bool (optional)
        Generate a trailing tail. Default: True.
    random_state : `~numpy.random.RandomState` (optional)
        To control random number generation.

    """
    def __init__(self, lead=True, trail=True, random_state=None):

        self._lead = int(lead)
        self._trail = int(trail)

        if random_state is None:
            random_state = np.random.RandomState()
        self.random_state = random_state

        if not self.lead and not self.trail:
            raise ValueError("You must generate either leading or trailing "
                             "tails (or both!)")

    cdef void get_rj_vj_R(self, CPotential *cpotential, double G,
                          double *prog_x, double *prog_v,
                          double prog_m, double t,
                          double *rj, double *vj, double[:, ::1] R): # outputs
        # NOTE: assuming ndim=3 throughout here
        cdef:
            int i
            double dist = norm(prog_x, 3)
            double L[3]
            double Lmag, Om, d2r

        # angular momentum vector, L, and |L|
        cross(prog_x, prog_v, &L[0])
        Lnorm = norm(&L[0], 3)

        # NOTE: R goes from non-rotating frame to rotating frame!!!
        for i in range(3):
            R[0, i] = prog_x[i] / dist
            R[2, i] = L[i] / Lnorm

        # Now compute jacobi radius and relative velocity at jacobi radius
        # Note: we re-use the L array as the "epsilon" array needed by d2_dr2
        Om = Lnorm / dist**2
        d2r = c_d2_dr2(cpotential, t, prog_x,
                       &L[0])
        rj[0] = (G * prog_m / (Om*Om - d2r)) ** (1/3.)
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


    cpdef _sample(self, potential,
                  double[:, ::1] prog_x, double[:, ::1] prog_v,
                  double[::1] prog_t, double[::1] prog_m, int[::1] nparticles):
        pass

    # ------------------------------------------------------------------------
    # Python-only:

    @property
    def lead(self):
        return self._lead

    @property
    def trail(self):
        return self._trail

    cpdef sample(self, prog_orbit, prog_mass, hamiltonian=None,
                 release_every=1, n_particles=1):
        """sample(prog_orbit, prog_mass, hamiltonian=None, release_every=1, n_particles=1)

        Generate stream particle initial conditions and initial times.

        This method is primarily meant to be used within the
        ``MockStreamGenerator``.

        Parameters
        ----------
        prog_orbit : `~gala.dynamics.Orbit`
            The orbit of the progenitor system.
        prog_mass : `~astropy.units.Quantity` [mass]
            The mass of the progenitor system, either a scalar quantity, or as
            an array with the same shape as the number of timesteps in the orbit
            to account for mass evolution.
        hamiltonian : `~gala.potential.Hamiltonian`
            The external potential and reference frame to numerically integrate
            orbits in.
        release_every : int (optional)
            Controls how often to release stream particles from each tail.
            Default: 1, meaning release particles at each timestep.
        n_particles : int, array_like (optional)
            If an integer, this controls the number of particles to release in
            each tail at each release timestep. Alternatively, you can pass in
            an array with the same shape as the number of timesteps to release
            bursts of particles at certain times (e.g., pericenter).

        Returns
        -------
        xyz : `~astropy.units.Quantity` [length]
            The initial positions for stream star particles.
        v_xyz : `~astropy.units.Quantity` [speed]
            The initial velocities for stream star particles.
        t1 : `~astropy.units.Quantity` [time]
            The initial times (i.e. times to start integrating from) for stream
            star particles.
        """

        if prog_orbit.hamiltonian is not None:
            H = prog_orbit.hamiltonian
        elif hamiltonian is not None:
            H = Hamiltonian(hamiltonian)
        else:
            raise ValueError('TODO')

        # TODO: if an orbit with non-static frame passed in, convert to static frame before generating
        static_frame = StaticFrame(H.units)
        frame = H.frame

        # TODO: we could catch this possible error and make it more specific
        prog_orbit_static = prog_orbit.to_frame(static_frame)

        # Coerce the input orbit into C-contiguous numpy arrays in the units of
        # the hamiltonian
        _units = H.units
        prog_x = np.ascontiguousarray(
            prog_orbit_static.xyz.decompose(_units).value.T)
        prog_v = np.ascontiguousarray(
            prog_orbit_static.v_xyz.decompose(_units).value.T)
        prog_t = prog_orbit_static.t.decompose(_units).value
        try:
            prog_m = prog_mass.decompose(_units).value
        except:
            raise TypeError("Input progenitor mass must be a Quantity object "
                            "with a decompose() method, e.g, an astropy "
                            "quantity.")

        if not isiterable(prog_m):
            prog_m = np.ones_like(prog_t) * prog_m

        if isiterable(n_particles):
            n_particles = np.array(n_particles).astype('i4')
            if not len(n_particles) == len(prog_t):
                raise ValueError('If passing in an array n_particles, its '
                                 'shape must match the number of timesteps in '
                                 'the progenitor orbit.')

        else:
            N = int(n_particles)
            n_particles = np.zeros_like(prog_t, dtype='i4')
            n_particles[::release_every] = N

        x, v, t1 = self._sample(H.potential, prog_x, prog_v,
                                prog_t, prog_m,
                                n_particles)

        # First out what particles are leading vs. trailing:
        lt = np.empty(len(t1), dtype='U1')
        i = 0
        for n in n_particles:
            if self._trail:
                lt[i:i+n] = 't'
                i += n

            if self._lead:
                lt[i:i+n] = 'l'
                i += n

        out = Orbit(pos=np.array(x).T * _units['length'],
                    vel=np.array(v).T * _units['length']/_units['time'],
                    t=np.array(t1) * _units['time'],
                    frame=static_frame)

        # Transform back to the input frame
        out = out.to_frame(frame)

        w0 = MockStream(pos=out.pos, vel=out.vel, frame=out.frame,
                        release_time=out.t, lead_trail=lt)

        return w0


@cython.embedsignature(True)
cdef class StreaklineStreamDF(BaseStreamDF):
    """A class for representing the "streakline" distribution function for
    generating stellar streams based on Kuepper et al. 2012
    https://ui.adsabs.harvard.edu/abs/2012MNRAS.420.2700K/abstract

    Parameters
    ----------
    lead : bool (optional)
        Generate a leading tail. Default: True.
    trail : bool (optional)
        Generate a trailing tail. Default: True.
    random_state : `~numpy.random.RandomState` (optional)
        To control random number generation.
    """

    cpdef _sample(self, potential,
                  double[:, ::1] prog_x, double[:, ::1] prog_v,
                  double[::1] prog_t, double[::1] prog_m, int[::1] nparticles):
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

            CPotential cpotential = (<CPotentialWrapper>(potential.c_instance)).cpotential
            double G = potential.G

        j = 0
        for i in range(ntimes):
            if prog_m[i] == 0:
                continue

            self.get_rj_vj_R(&cpotential, G,
                             &prog_x[i, 0], &prog_v[i, 0], prog_m[i], prog_t[i],
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


@cython.embedsignature(True)
cdef class FardalStreamDF(BaseStreamDF):
    """A class for representing the Fardal+2015 distribution function for
    generating stellar streams based on Fardal et al. 2015
    https://ui.adsabs.harvard.edu/abs/2015MNRAS.452..301F/abstract

    Parameters
    ----------
    gala_modified : bool (optional)
        If True, use the modified version of the Fardal method parameters used in Gala. If you would like to use the exact parameters from Fardal+2015, set this to False. Default: True.
    lead : bool (optional)
        Generate a leading tail. Default: True.
    trail : bool (optional)
        Generate a trailing tail. Default: True.
    random_state : `~numpy.random.RandomState` (optional)
        To control random number generation.
    """
    def __init__(
        self, gala_modified=None, lead=True, trail=True, random_state=None
    ):
        super().__init__(lead=lead, trail=trail, random_state=random_state)

        if gala_modified is None:
            from gala.util import GalaDeprecationWarning
            import warnings
            msg = (
                "The parameter values of the FardalStreamDF have been updated (fixed) "
                "to match the parameter values in the final published version of "
                "Fardal+2015. For now, this class uses the Gala modified parameter "
                "values that have been adopted over the last several years in Gala. "
                "In the future, the default behavior of this class will use the "
                "Fardal+2015 parameter values instead, breaking backwards "
                "compatibility for mock stream simulations. To use the Fardal+2015 "
                "parameters now, set gala_modified=False. To continue to use the Gala "
                "modified parameter values, set gala_modified=True."
            )
            warnings.warn(msg, GalaDeprecationWarning)
            gala_modified = True

        self._gala_modified = int(gala_modified)


    cpdef _sample(self, potential,
                  double[:, ::1] prog_x, double[:, ::1] prog_v,
                  double[::1] prog_t, double[::1] prog_m, int[::1] nparticles):
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

            CPotential cpotential = (<CPotentialWrapper>(potential.c_instance)).cpotential
            double G = potential.G

        # TODO: support computing this, which requires knowing the peri/apo and values
        # of Om**2 - d2Phi/dr2 at those points...
        # kvt_fardal = min(0.15 * self.f_t**2 * Racc**(2/3), 0.4)
        kvt_fardal = 0.4

        k_mean[0] = 2. # R
        k_disp[0] = 0.5 if self._gala_modified else 0.4

        k_mean[2] = 0. # z
        k_disp[2] = 0.5

        k_mean[4] = 0.3 # vt
        k_disp[4] = 0.5 if self._gala_modified else kvt_fardal

        k_mean[5] = 0. # vz
        k_disp[5] = 0.5

        j = 0
        for i in range(ntimes):
            if prog_m[i] == 0:
                continue

            self.get_rj_vj_R(&cpotential, G,
                             &prog_x[i, 0], &prog_v[i, 0], prog_m[i], prog_t[i],
                             &rj, &vj, R)  # outputs

            # Trailing tail
            if self._trail == 1:
                for k in range(nparticles[i]):
                    kx = self.random_state.normal(k_mean[0], k_disp[0])
                    tmp_x[0] = kx * rj
                    tmp_x[2] = self.random_state.normal(k_mean[2], k_disp[2]) * rj
                    tmp_v[1] = self.random_state.normal(k_mean[4], k_disp[4]) * vj
                    if self._gala_modified:  # for backwards compatibility
                        tmp_v[1] *= kx
                    tmp_v[2] = self.random_state.normal(k_mean[5], k_disp[5]) * vj
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
                    kx = self.random_state.normal(k_mean[0], k_disp[0])
                    tmp_x[0] = kx * -rj
                    tmp_x[2] = self.random_state.normal(k_mean[2], k_disp[2]) * -rj
                    tmp_v[1] = self.random_state.normal(k_mean[4], k_disp[4]) * -vj
                    if self._gala_modified:  # for backwards compatibility
                        tmp_v[1] *= kx
                    tmp_v[2] = self.random_state.normal(k_mean[5], k_disp[5]) * -vj
                    particle_t1[j+k] = prog_t[i]

                    self.transform_from_sat(R,
                                            &tmp_x[0], &tmp_v[0],
                                            &prog_x[i, 0], &prog_v[i, 0],
                                            &particle_x[j+k, 0],
                                            &particle_v[j+k, 0])

                j += nparticles[i]

        return particle_x, particle_v, particle_t1


@cython.embedsignature(True)
cdef class LagrangeCloudStreamDF(BaseStreamDF):
    """A class for representing the Lagrange Cloud Stripping distribution
    function for generating stellar streams. This df is based on Gibbons et al.
    2014 https://ui.adsabs.harvard.edu/abs/2014MNRAS.445.3788G/abstract
    but has since been modified by, e.g., Erkal et al. 2019
    https://ui.adsabs.harvard.edu/abs/2019MNRAS.487.2685E/abstract .

    Parameters
    ----------
    v_disp : `~astropy.units.Quantity` [speed]
        The velocity dispersion of the released particles.
    lead : bool (optional)
        Generate a leading tail. Default: True.
    trail : bool (optional)
        Generate a trailing tail. Default: True.
    random_state : `~numpy.random.RandomState` (optional)
        To control random number generation.
    """

    cdef public object v_disp

    @u.quantity_input(v_disp=u.km/u.s)
    def __init__(self, v_disp, lead=True, trail=True, random_state=None):
        super().__init__(lead=lead, trail=trail, random_state=random_state)

        self.v_disp = v_disp

    cpdef _sample(self, potential,
                  double[:, ::1] prog_x, double[:, ::1] prog_v,
                  double[::1] prog_t, double[::1] prog_m, int[::1] nparticles):
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

            CPotential cpotential = (<CPotentialWrapper>(potential.c_instance)).cpotential
            double G = potential.G
            double _v_disp = self.v_disp.decompose(potential.units).value

        j = 0
        for i in range(ntimes):
            if prog_m[i] == 0:
                continue

            self.get_rj_vj_R(&cpotential, G,
                             &prog_x[i, 0], &prog_v[i, 0], prog_m[i], prog_t[i],
                             &rj, &vj, R) # outputs

            # Trailing tail
            if self._trail == 1:
                for k in range(nparticles[i]):
                    tmp_x[0] = rj
                    tmp_v[0] = self.random_state.normal(0, _v_disp)
                    tmp_v[1] = self.random_state.normal(0, _v_disp)
                    tmp_v[2] = self.random_state.normal(0, _v_disp)
                    particle_t1[j + k] = prog_t[i]

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
                    tmp_v[0] = self.random_state.normal(0, _v_disp)
                    tmp_v[1] = self.random_state.normal(0, _v_disp)
                    tmp_v[2] = self.random_state.normal(0, _v_disp)
                    particle_t1[j + k] = prog_t[i]

                    self.transform_from_sat(R,
                                            &tmp_x[0], &tmp_v[0],
                                            &prog_x[i, 0], &prog_v[i, 0],
                                            &particle_x[j+k, 0],
                                            &particle_v[j+k, 0])

                j += nparticles[i]

        return particle_x, particle_v, particle_t1


@cython.embedsignature(True)
cdef class ChenStreamDF(BaseStreamDF):
    """A class for representing the Chen+2024 distribution function for
    generating stellar streams based on Chen et al. 2024
    https://ui.adsabs.harvard.edu/abs/2024arXiv240801496C/abstract

    Parameters
    ----------
    lead : bool (optional)
        Generate a leading tail. Default: True.
    trail : bool (optional)
        Generate a trailing tail. Default: True.
    random_state : `~numpy.random.RandomState` (optional)
        To control random number generation.
    """
    def __init__(
        self, lead=True, trail=True, random_state=None
    ):
        super().__init__(lead=lead, trail=trail, random_state=random_state)


    cpdef _sample(self, potential,
                  double[:, ::1] prog_x, double[:, ::1] prog_v,
                  double[::1] prog_t, double[::1] prog_m, int[::1] nparticles):
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

            # for Chen method:
            double Dr
            double Dv
            double[::1] posvel = np.zeros(6)
            double[::1] mean = np.zeros(6)
            double[:, ::1] cov = np.zeros((6, 6))

            CPotential cpotential = (<CPotentialWrapper>(potential.c_instance)).cpotential
            double G = potential.G

        mean[0] = 1.6    # r
        cov[0, 0] = 0.1225

        mean[1] = -30.   # phi
        cov[1, 1] = 529.

        mean[2] = 0.     # theta
        cov[2, 2] = 144.

        mean[3] = 1.     # v
        cov[3, 3] = 0.

        mean[4] = 20.    # alpha
        cov[4, 4] = 400.

        mean[5] = 0.     # beta
        cov[5, 5] = 484.

        cov[0, 4] = -4.9 # covariance between r and alpha
        cov[4, 0] = -4.9

        j = 0
        for i in range(ntimes):
            if prog_m[i] == 0:
                continue

            self.get_rj_vj_R(&cpotential, G,
                             &prog_x[i, 0], &prog_v[i, 0], prog_m[i], prog_t[i],
                             &rj, &vj, R)  # outputs

            # trailing tail
            if self._trail == 1:
                for k in range(nparticles[i]):
                    # calculate the ejection position and velocity
                    posvel = self.random_state.multivariate_normal(mean, cov)

                    Dr = posvel[0] * rj
                    Dv = posvel[3] * sqrt(2*G*prog_m[i]/Dr) # escape velocity

                    # convert degrees to radians
                    posvel[1] = posvel[1] * (M_PI/180)
                    posvel[2] = posvel[2] * (M_PI/180)
                    posvel[4] = posvel[4] * (M_PI/180)
                    posvel[5] = posvel[5] * (M_PI/180)

                    tmp_x[0] = Dr*cos(posvel[2])*cos(posvel[1])
                    tmp_x[1] = Dr*cos(posvel[2])*sin(posvel[1])
                    tmp_x[2] = Dr*sin(posvel[2])

                    tmp_v[0] = Dv*cos(posvel[5])*cos(posvel[4])
                    tmp_v[1] = Dv*cos(posvel[5])*sin(posvel[4])
                    tmp_v[2] = Dv*sin(posvel[5])

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
                    # calculate the ejection position and velocity
                    posvel = self.random_state.multivariate_normal(mean, cov)

                    Dr = posvel[0] * rj
                    Dv = posvel[3] * sqrt(2*G*prog_m[i]/Dr) # escape velocity

                    # convert degrees to radians
                    posvel[1] = posvel[1] * (M_PI/180) + M_PI
                    posvel[2] = posvel[2] * (M_PI/180)
                    posvel[4] = posvel[4] * (M_PI/180) + M_PI
                    posvel[5] = posvel[5] * (M_PI/180)

                    tmp_x[0] = Dr*cos(posvel[2])*cos(posvel[1])
                    tmp_x[1] = Dr*cos(posvel[2])*sin(posvel[1])
                    tmp_x[2] = Dr*sin(posvel[2])

                    tmp_v[0] = Dv*cos(posvel[5])*cos(posvel[4])
                    tmp_v[1] = Dv*cos(posvel[5])*sin(posvel[4])
                    tmp_v[2] = Dv*sin(posvel[5])

                    particle_t1[j+k] = prog_t[i]

                    self.transform_from_sat(R,
                                            &tmp_x[0], &tmp_v[0],
                                            &prog_x[i, 0], &prog_v[i, 0],
                                            &particle_x[j+k, 0],
                                            &particle_v[j+k, 0])
                
                j += nparticles[i]

        return particle_x, particle_v, particle_t1