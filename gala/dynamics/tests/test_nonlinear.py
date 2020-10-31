# Third-party
import numpy as np

# Project
from ... import potential as gp
from ...potential import Hamiltonian
from ..nonlinear import lyapunov_max, fast_lyapunov_max, surface_of_section
from ...integrate import DOPRI853Integrator
from ...units import galactic


class TestForcedPendulum(object):

    def setup(self):

        def F(t, x, A, omega_d):
            q, p = x
            return np.array([p, -np.sin(q) + A*np.cos(omega_d*t)])

        # initial conditions and parameter choices for chaotic / regular pendulum
        self.regular_w0 = np.array([1., 0.])
        self.regular_par = (0.055, 0.7)
        self.regular_integrator = DOPRI853Integrator(F, func_args=self.regular_par)

        self.chaotic_w0 = np.array([3., 0.])
        self.chaotic_par = (0.07, 0.75)
        self.chaotic_integrator = DOPRI853Integrator(F, func_args=self.chaotic_par)

    def test_lyapunov_max(self, tmpdir):
        n_steps = 20000
        dt = 1.
        n_steps_per_pullback = 10
        d0 = 1e-5
        noffset = 2

        regular_LEs, regular_orbit = lyapunov_max(self.regular_w0, self.regular_integrator,
                                                  dt=dt, n_steps=n_steps,
                                                  d0=d0, n_steps_per_pullback=n_steps_per_pullback,
                                                  noffset_orbits=noffset)

        regular_LEs = np.mean(regular_LEs, axis=1)
        assert regular_LEs[-1] < 1E-3

        chaotic_LEs, chaotic_orbit = lyapunov_max(self.chaotic_w0, self.chaotic_integrator,
                                                  dt=dt, n_steps=n_steps,
                                                  d0=d0, n_steps_per_pullback=n_steps_per_pullback,
                                                  noffset_orbits=noffset)
        chaotic_LEs = np.mean(chaotic_LEs, axis=1)
        assert chaotic_LEs[-1] > 1E-2

        # pl.figure()
        # pl.loglog(regular_LEs, marker='')
        # pl.savefig(os.path.join(str(tmpdir),"pend_regular.png"))

        # pl.figure()
        # pl.plot(t, regular_ws[:, 0], marker='')
        # pl.savefig(os.path.join(str(tmpdir),"pend_orbit_regular.png"))

        # pl.figure()
        # pl.loglog(chaotic_LEs, marker='')
        # pl.savefig(os.path.join(str(tmpdir),"pend_chaotic.png"))

        # pl.figure()
        # pl.plot(t, chaotic_ws[:, 0], marker='')
        # pl.savefig(os.path.join(str(tmpdir),"pend_orbit_chaotic.png"))

        # pl.close('all')

# --------------------------------------------------------------------


class HenonHeilesBase(object):

    def potential(self, w, A, B, C, D):
        x, y = w[:2]
        term1 = 0.5*(A*x**2 + B*y**2)
        term2 = D*x**2*y - C/3.*y**3
        return term1 + term2

    def acceleration(self, w, A, B, C, D):
        x, y = w[:2]
        ax = -(A*x + 2*D*x*y)
        ay = -(B*y + D*x*x - C*y*y)
        return np.array([ax, ay])

    def jerk(self, w, A, B, C, D):
        x, y = w[:2]
        dx, dy = w[4:6]

        dax = -(A+2*D*y)*dx - 2*D*x*dy
        day = -2*D*x*dx - (B-2*C*y)*dy

        return np.array([dax, day])

    def F_max(self, t, w, *args):
        x, y, px, py = w
        term1 = np.array([px, py])
        term2 = self.acceleration(w, *args)
        return np.vstack((term1, term2))

    def setup(self):
        # parameter choices
        self.par = (1., 1., 1., 1.)
        self.n_steps = 2000
        self.dt = 2.

    def test_integrate_orbit(self, tmpdir):
        integrator = DOPRI853Integrator(self.F_max, func_args=self.par)
        orbit = integrator.run(self.w0, dt=self.dt, n_steps=self.n_steps)

    def test_lyapunov_max(self, tmpdir):
        n_steps_per_pullback = 10
        d0 = 1e-5
        noffset = 2

        integrator = DOPRI853Integrator(self.F_max, func_args=self.par)
        lyap, orbit = lyapunov_max(self.w0, integrator,
                                   dt=self.dt, n_steps=self.n_steps,
                                   d0=d0, noffset_orbits=noffset,
                                   n_steps_per_pullback=n_steps_per_pullback)
        lyap = np.mean(lyap, axis=1)

        # pl.clf()
        # pl.loglog(lyap, marker='')
        # pl.savefig(os.path.join(str(tmpdir),"hh_lyap_max_{}.png".format(self.__class__.__name__)))

        # pl.clf()
        # pl.plot(ws[..., 0], ws[..., 1], marker='')
        # pl.savefig(os.path.join(str(tmpdir),"hh_orbit_lyap_max_{}.png".format(self.__class__.__name__)))


# initial conditions from LP-VI documentation:
class TestHenonHeilesStablePeriodic(HenonHeilesBase):
    def setup(self):
        super().setup()
        self.w0 = np.array([0., 0.295456, 0.407308431, 0.])
        self.check = lambda x: x < 1E-3


class TestHenonHeilesStableQuasi1(HenonHeilesBase):
    def setup(self):
        super().setup()
        self.w0 = np.array([0., 0.483, 0.27898039, 0.])
        self.check = lambda x: x < 2E-3


class TestHenonHeilesStableQuasi2(HenonHeilesBase):
    def setup(self):
        super().setup()
        self.w0 = np.array([0., 0.46912, 0.291124891, 0.])
        self.check = lambda x: x < 2E-3


class TestHenonHeilesStableChaos1(HenonHeilesBase):
    def setup(self):
        super().setup()
        self.w0 = np.array([0., 0.509, 0.254624859, 0.])
        self.check = lambda x: x > 2E-3


class TestHenonHeilesStableChaos2(HenonHeilesBase):
    def setup(self):
        super().setup()
        self.w0 = np.array([0., 0.56, 0.164113781, 0.112])
        self.check = lambda x: x > 1E-2

# --------------------------------------------------------------------


class TestLogarithmic(object):

    def setup(self):

        # set the potential
        potential = gp.LogarithmicPotential(v_c=np.sqrt(2), r_h=0.1,
                                            q1=1., q2=0.9, q3=1.,
                                            units=galactic)
        self.hamiltonian = Hamiltonian(potential)

        # see figure 1 from Papaphillipou & Laskar
        x0 = -0.01
        X0 = -0.2
        y0 = 0.
        E0 = -0.4059
        Y0 = np.sqrt(E0 - self.hamiltonian.potential.energy([x0, y0, 0.]).value)
        chaotic_w0 = [x0, y0, 0., X0, Y0, 0.]

        # initial conditions from LP-VI documentation:
        self.w0s = np.array([[0.49, 0., 0., 1.3156, 0.4788, 0.],  # regular
                             chaotic_w0])  # chaotic

        self.n_steps = 25000
        self.dt = 0.004

    def test_fast_lyapunov_max(self, tmpdir):
        n_steps_per_pullback = 10
        d0 = 1e-5
        noffset = 2

        for ii, w0 in enumerate(self.w0s):
            print(ii, w0)
            lyap, orbit = fast_lyapunov_max(w0, self.hamiltonian,
                                            dt=self.dt, n_steps=self.n_steps,
                                            d0=d0, noffset_orbits=noffset,
                                            n_steps_per_pullback=n_steps_per_pullback)
            lyap = np.mean(lyap, axis=1)

            # also just integrate the orbit to compare dE scaling
            orbit2 = self.hamiltonian.integrate_orbit(w0, dt=self.dt, n_steps=self.n_steps,
                                                      Integrator=DOPRI853Integrator)

            # lyapunov exp
            # pl.figure()
            # pl.loglog(lyap, marker='')
            # pl.savefig(os.path.join(str(tmpdir),"log_lyap_max_{}.png".format(ii)))

            # energy conservation
            E = orbit[:, 0].energy().value  # returns 3 orbits
            dE = np.abs(E[1:] - E[0])

            E = orbit2.energy().value
            dE_ww = np.abs(E[1:] - E[0])

            # import matplotlib.pyplot as plt
            # plt.semilogy(dE, marker='')
            # plt.semilogy(dE_ww, marker='')

            # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            # axes[0].plot(orbit.pos[0, :, 0], orbit.pos[1, :, 0], marker='') # ignore offset orbits
            # axes[1].plot(orbit2.pos[0], orbit2.pos[1], marker='')
            # fig.savefig(os.path.join(str(tmpdir),"log_orbit_lyap_max_{}.png".format(ii)))

            # plt.show()

            assert np.allclose(dE_ww[-100:], dE[-100:], rtol=1E-1)

    def test_compare_fast(self, tmpdir):
        n_steps_per_pullback = 10
        d0 = 1e-5
        noffset = 2

        def F(t, w):
            w_T = np.ascontiguousarray(w.T)
            return self.hamiltonian._gradient(w_T, np.array([t])).T

        integrator = DOPRI853Integrator(F)
        for ii, w0 in enumerate(self.w0s):
            lyap1, orbit1 = fast_lyapunov_max(w0, self.hamiltonian,
                                              dt=self.dt, n_steps=self.n_steps//8,
                                              d0=d0, noffset_orbits=noffset,
                                              n_steps_per_pullback=n_steps_per_pullback)
            lyap1 = np.mean(lyap1, axis=1)

            # check energy conservation
            E = orbit1.energy().value
            dE_fast = np.abs(E[1:] - E[0])
            assert np.all(dE_fast[:, 0] < 1E-10)

            lyap2, orbit2 = lyapunov_max(w0.copy(), integrator,
                                         dt=self.dt, n_steps=self.n_steps//8,
                                         d0=d0, noffset_orbits=noffset,
                                         n_steps_per_pullback=n_steps_per_pullback,
                                         units=self.hamiltonian.units)
            lyap2 = np.mean(lyap2, axis=1)

            # check energy conservation
            E = orbit2.energy(self.hamiltonian).value
            dE_slow = np.abs(E[1:] - E[0])

            if not np.all(dE_slow[:, 0] < 1E-10):
                import matplotlib.pyplot as plt

                plt.figure()
                plt.plot(orbit2.pos[0, -128:, 0], orbit2.pos[1, -128:, 0], marker='.')
                plt.plot(orbit2.pos[0, -128:, 0], orbit2.pos[2, -128:, 0], marker='.')

                plt.figure()
                plt.semilogy(dE_slow[:, 0], marker='.')

                plt.show()

            assert np.all(dE_slow[:, 0] < 1E-10)

            # plots
            # import matplotlib.pyplot as plt

            # plt.figure()
            # plt.loglog(orbit1.t[1:-10:10], lyap1, marker='')
            # plt.loglog(orbit2.t[1:-10:10], lyap2, marker='')
            # plt.savefig(os.path.join(str(tmpdir),"log_lyap_compare_{}.png".format(ii)))

            # plt.figure()
            # plt.semilogy(dE_fast[:, 0], marker='')
            # plt.semilogy(dE_slow[:, 0], marker='')
            # # plt.savefig(os.path.join(str(tmpdir),"log_dE_{}.png".format(ii)))

            # fig, axes = plt.subplots(1, 2, figsize=(6, 6))
            # axes[0].plot(orbit1.pos[0, :, 0], orbit1.pos[1, :, 0],
            #              marker='.', linestyle='none', alpha=0.1)
            # axes[1].plot(orbit2.pos[0, :, 0], orbit2.pos[1, :, 0],
            #              marker='.', linestyle='none', alpha=0.1)
            # plt.savefig(os.path.join(str(tmpdir),"log_orbit_lyap_max_{}.png".format(ii)))

            # plt.show()
            # plt.close('all')


def test_surface_of_section():
    from ...potential import LogarithmicPotential
    from ...units import galactic

    pot = LogarithmicPotential(v_c=1., r_h=1.,
                               q1=1., q2=0.9, q3=0.8,
                               units=galactic)

    w0 = np.array([0., 0.8, 0., 1., 0., 0.])
    orbit = Hamiltonian(pot).integrate_orbit(w0, dt=0.02, n_steps=100_000)
    sos = surface_of_section(orbit, constant_idx=1)
    sos_cyl = surface_of_section(orbit.cylindrical, constant_idx=1)
