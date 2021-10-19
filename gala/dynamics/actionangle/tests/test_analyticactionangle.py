# Third-party
import numpy as np
import astropy.units as u

# Project
from gala.dynamics.actionangle import (
    isochrone_xv_to_aa,
    isochrone_aa_to_xv,
    harmonic_oscillator_xv_to_aa,
)
from gala.logging import logger
from gala.potential import (
    IsochronePotential,
    HarmonicOscillatorPotential,
    Hamiltonian,
)
from gala.units import galactic
from gala.util import assert_angles_allclose
from gala.dynamics.actionangle._genfunc import toy_potentials


class TestIsochrone(object):
    def setup(self):
        logger.info("======== Isochrone ========")
        self.N = 100
        np.random.seed(42)
        x = np.random.uniform(-10.0, 10.0, size=(3, self.N))
        v = np.random.uniform(-1.0, 1.0, size=(3, self.N)) / 33.0
        w0 = np.vstack((x, v))

        self.potential = IsochronePotential(units=galactic, m=1.0e11, b=5.0)
        H = Hamiltonian(self.potential)
        self.w = H.integrate_orbit(w0, dt=0.1, n_steps=10000)
        self.w = self.w[::10]

    def test(self):
        # TODO: doesn't need to be a loop?
        for n in range(self.N):
            logger.debug("Orbit {}".format(n))

            actions, angles, freqs = isochrone_xv_to_aa(
                self.w[:, n], self.potential
            )

            for i in range(3):
                assert u.allclose(actions[i, 1:], actions[i, 0], rtol=1e-5)

            # Compare to genfunc
            x = self.w.xyz[..., n]
            v = self.w.v_xyz[..., n]
            s_w = np.vstack((x.to_value(u.kpc), v.to_value(u.km / u.s)))
            m = self.potential.parameters["m"].value / 1e11
            b = self.potential.parameters["b"].value
            aa = np.array(
                [
                    toy_potentials.angact_iso(s_w[:, i].T, params=(m, b))
                    for i in range(s_w.shape[1])
                ]
            )
            s_actions = aa[:, :3] * u.km / u.s * u.kpc
            s_angles = aa[:, 3:] * u.rad

            assert u.allclose(actions, s_actions.T, rtol=1e-8)
            assert_angles_allclose(angles, s_angles.T, rtol=1e-8)

            w_rt = isochrone_aa_to_xv(actions, angles, self.potential)

            assert u.allclose(x, w_rt.xyz, atol=1E-10 * u.kpc)
            assert u.allclose(v, w_rt.v_xyz, atol=1E-10 * u.km/u.s)


class TestHarmonicOscillator(object):
    def setup(self):
        logger.info("======== Harmonic Oscillator ========")
        self.N = 100
        np.random.seed(42)
        x = np.random.uniform(-10.0, 10.0, size=(3, self.N))
        v = np.random.uniform(-1.0, 1.0, size=(3, self.N)) / 33.0
        w0 = np.vstack((x, v))

        self.potential = HarmonicOscillatorPotential(
            omega=np.array([0.013, 0.02, 0.005]), units=galactic
        )
        H = Hamiltonian(self.potential)
        self.w = H.integrate_orbit(w0, dt=0.1, n_steps=10000)
        self.w = self.w[::10]

    def test(self):
        """
        !!!!! NOTE !!!!!
        For Harmonic Oscillator, Sanders' code works for the units I use...
        """
        for n in range(self.N):
            logger.debug("Orbit {}".format(n))

            actions, angles, freq = harmonic_oscillator_xv_to_aa(
                self.w[:, n], self.potential
            )
            actions = actions.value
            angles = angles.value

            for i in range(3):
                assert np.allclose(actions[i, 1:], actions[i, 0], rtol=1e-5)

            # Compare to genfunc
            x = self.w.xyz.value[..., n]
            v = self.w.v_xyz.value[..., n]
            s_w = np.vstack((x, v))
            omega = self.potential.parameters["omega"].value
            aa = np.array(
                [
                    toy_potentials.angact_ho(s_w[:, i].T, omega=omega)
                    for i in range(s_w.shape[1])
                ]
            )
            s_actions = aa[:, :3]
            s_angles = aa[:, 3:]

            assert np.allclose(actions, s_actions.T, rtol=1e-8)
            assert_angles_allclose(angles, s_angles.T, rtol=1e-8)

            # test roundtrip
            # x2, v2 = harmonic_oscillator_aa_to_xv(actions, angles, self.potential)
            # TODO: transform back??
