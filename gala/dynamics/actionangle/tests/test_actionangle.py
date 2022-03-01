""" Test action-angle stuff """

# Standard library
import logging
import warnings

# Third-party
import astropy.units as u
import numpy as np
from gala.logging import logger
from scipy.linalg import solve
import pytest

# Project
from gala.integrate import DOPRI853Integrator
from gala.potential import (
    IsochronePotential,
    HarmonicOscillatorPotential,
    LeeSutoTriaxialNFWPotential,
    Hamiltonian,
)
from gala.units import galactic
from gala.dynamics.actionangle import (
    fit_isochrone,
    fit_harmonic_oscillator,
    fit_toy_potential,
    check_angle_sampling,
    find_actions,
    generate_n_vectors,
)
from gala.dynamics.actionangle._genfunc import genfunc_3d, solver
from .helpers import sanders_nvecs, sanders_act_ang_freq, isotropic_w0

logger.setLevel(logging.DEBUG)


def test_generate_n_vectors():
    # test against Sanders' method
    nvecs = generate_n_vectors(N_max=6, dx=2, dy=2, dz=2)
    nvecs_sanders = sanders_nvecs(N_max=6, dx=2, dy=2, dz=2)
    assert np.all(nvecs == nvecs_sanders)

    nvecs = generate_n_vectors(N_max=6, dx=1, dy=1, dz=1)
    nvecs_sanders = sanders_nvecs(N_max=6, dx=1, dy=1, dz=1)
    assert np.all(nvecs == nvecs_sanders)


def test_fit_isochrone():
    # integrate orbit in Isochrone potential, then try to recover it
    true_m = 2.81e11
    true_b = 11.0
    potential = IsochronePotential(m=true_m, b=true_b, units=galactic)
    H = Hamiltonian(potential)
    orbit = H.integrate_orbit([15.0, 0, 0, 0, 0.2, 0], dt=2.0, n_steps=10000)

    fit_potential = fit_isochrone(orbit)
    m, b = (
        fit_potential.parameters["m"].value,
        fit_potential.parameters["b"].value,
    )
    assert np.allclose(m, true_m, rtol=1e-2)
    assert np.allclose(b, true_b, rtol=1e-2)


def test_fit_harmonic_oscillator():
    # integrate orbit in harmonic oscillator potential, then try to recover it
    true_omegas = np.array([0.011, 0.032, 0.045])
    potential = HarmonicOscillatorPotential(omega=true_omegas, units=galactic)
    H = Hamiltonian(potential)
    orbit = H.integrate_orbit([15.0, 1, 2, 0, 0, 0], dt=2.0, n_steps=10000)

    fit_potential = fit_harmonic_oscillator(orbit)
    omegas = fit_potential.parameters["omega"].value
    assert np.allclose(omegas, true_omegas, rtol=1e-2)


def test_fit_toy_potential():
    # integrate orbit in both toy potentials, make sure correct one is chosen
    true_m = 2.81e11
    true_b = 11.0
    true_potential = IsochronePotential(m=true_m, b=true_b, units=galactic)
    H = Hamiltonian(true_potential)
    orbit = H.integrate_orbit([15.0, 0, 0, 0, 0.2, 0], dt=2.0, n_steps=10000)

    potential = fit_toy_potential(orbit)
    for k, v in true_potential.parameters.items():
        assert u.allclose(v, potential.parameters[k], rtol=1e-2)

    # -----------------------------------------------------------------
    true_omegas = np.array([0.011, 0.032, 0.045])
    true_potential = HarmonicOscillatorPotential(
        omega=true_omegas, units=galactic
    )
    H = Hamiltonian(true_potential)
    orbit = H.integrate_orbit([15.0, 1, 2, 0, 0, 0], dt=2.0, n_steps=10000)

    potential = fit_toy_potential(orbit)

    assert u.allclose(
        potential.parameters["omega"],
        true_potential.parameters["omega"],
        rtol=1e-2,
    )


def test_check_angle_sampling():

    # frequencies
    omegas = np.array([0.21, 0.3421, 0.4968])

    # integer vectors
    nvecs = generate_n_vectors(N_max=6)

    # loop over times with known failures:
    #   - first one fails needing longer integration time
    #   - second one fails needing finer sampling
    for i, t in enumerate(
        [np.linspace(0, 50, 500), np.linspace(0, 8000, 8000)]
    ):
        # periods = 2*np.pi/omegas
        # print("Periods:", periods)
        # print("N periods:", t.max() / periods)

        angles = t[np.newaxis] * omegas[:, np.newaxis]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            checks, failures = check_angle_sampling(nvecs, angles)

        assert np.all(failures == i)


class ActionsBase:

    def test_classify(self):
        # my classify
        orb_type = self.orbit.circulation()

        # compare to Sanders'
        for j in range(self.N):
            sdrs = genfunc_3d.assess_angmom(self.w[..., j].T)
            logger.debug("APW: {}, Sanders: {}".format(orb_type[:, j], sdrs))
            assert np.all(orb_type[:, j] == sdrs)

    def test_actions(self):
        # t = self.t[::10]
        t = self.t

        N_max = 6
        for n in range(self.N):
            print("\n\n")
            print(
                "======================= Orbit {} =======================".format(
                    n
                )
            )
            # w = self.w[:, ::10, n]
            w = self.w[..., n]
            orb = self.orbit[:, n]
            circ = orb.circulation()

            # get values from Sanders' code
            print("Computing actions from genfunc...")
            s_actions, s_angles, s_freqs, toy_potential = sanders_act_ang_freq(
                t, w, circ, N_max=N_max
            )

            print("Computing actions with gala...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                ret = find_actions(
                    orb, N_max=N_max, toy_potential=toy_potential
                )
            actions = ret["actions"]
            angles = ret["angles"]
            freqs = ret["freqs"]

            print("Action ratio: {}".format(actions / s_actions))
            print("Angle ratio: {}".format(angles / s_angles))
            print("Freq ratio: {}".format(freqs / s_freqs))

            assert np.allclose(actions.value, s_actions, rtol=1e-5)
            assert np.allclose(angles.value, s_angles, rtol=1e-5)
            assert np.allclose(freqs.value, s_freqs, rtol=1e-5)

            # logger.debug("Plotting orbit...")
            # fig = plot_orbits(w, marker='.', alpha=0.2, linestyle='none')
            # fig.savefig(str(self.plot_path.join("orbit_{}.png".format(n))))

            # fig = plot_angles(t, angles, freqs)
            # fig.savefig(str(self.plot_path.join("angles_{}.png".format(n))))

            # fig = plot_angles(t, s_angles, s_freqs)
            # fig.savefig(str(self.plot_path.join("angles_sanders_{}.png".format(n))))

            # plt.close('all')

            # print("Plots saved at:", self.plot_path)


class TestActions(ActionsBase):

    @pytest.fixture(autouse=True)
    def setup(self, tmpdir):
        self.plot_path = tmpdir.mkdir("normal")

        self.units = galactic
        self.potential = LeeSutoTriaxialNFWPotential(
            v_c=0.2, r_s=20.0, a=1.0, b=0.77, c=0.55, units=galactic
        )
        self.N = 8
        np.random.seed(42)
        w0 = isotropic_w0(N=self.N)
        n_steps = 20000

        # integrate orbits
        H = Hamiltonian(self.potential)
        orbit = H.integrate_orbit(
            w0, dt=2.0, n_steps=n_steps, Integrator=DOPRI853Integrator
        )
        self.orbit = orbit
        self.t = orbit.t.value
        self.w = orbit.w()


def test_compare_action_prepare():

    from gala.dynamics.actionangle.actionangle_o2gf import (
        _action_prepare, _angle_prepare)

    logger.setLevel(logging.ERROR)
    AA = np.random.uniform(0.0, 100.0, size=(1000, 6))
    t = np.linspace(0.0, 100.0, 1000)

    act_san, n_vectors = solver.solver(AA, N_max=6, symNx=2)
    A2, b2, n = _action_prepare(AA.T, N_max=6, dx=2, dy=2, dz=2)
    act_apw = np.array(solve(A2, b2))

    ang_san = solver.angle_solver(AA, t, N_max=6, symNx=2, sign=1)
    A2, b2, n = _angle_prepare(AA.T, t, N_max=6, dx=2, dy=2, dz=2)
    ang_apw = np.array(solve(A2, b2))

    assert np.allclose(act_apw, act_san)
    # assert np.allclose(ang_apw, ang_san)

    # TODO: this could be critical -- why don't our angles agree?


def test_regression_113():
    """Test that fit_isochrone succeeds for a variety of orbits. See issue:
    https://github.com/adrn/gala/issues/113
    """
    from gala.potential import MilkyWayPotential, Hamiltonian
    from gala.dynamics import PhaseSpacePosition

    pot = MilkyWayPotential()

    dt = 0.01
    n_steps = 50000

    rvec = [0.3, 0, 0] * u.kpc
    vinit = pot.circular_velocity(rvec)[0].to(u.km / u.s).value
    vvec = [0, vinit * np.cos(0.01), vinit * np.sin(0.01)] * u.km / u.s
    vvec = 0.999 * vvec

    ics = PhaseSpacePosition(pos=rvec, vel=vvec)
    H = Hamiltonian(pot)
    orbit = H.integrate_orbit(ics, dt=dt, n_steps=n_steps)
    toy_potential = fit_isochrone(orbit)

    m = toy_potential.parameters["m"].value
    b = toy_potential.parameters["b"].value
    assert np.log10(m) > 11 and np.log10(m) < 12
    assert np.log10(b) > 0 and np.log10(b) < 1

    # try again with Nelder-Mead
    toy_potential = fit_isochrone(
        orbit, minimize_kwargs=dict(method="Nelder-Mead")
    )

    m = toy_potential.parameters["m"].value
    b = toy_potential.parameters["b"].value
    assert np.log10(m) > 11 and np.log10(m) < 12
    assert np.log10(b) > 0 and np.log10(b) < 1
