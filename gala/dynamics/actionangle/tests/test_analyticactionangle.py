import astropy.units as u
import numpy as np

from gala.dynamics.actionangle import (
    harmonic_oscillator_xv_to_aa,
    isochrone_aa_to_xv,
    isochrone_xv_to_aa,
)
from gala.dynamics.actionangle._genfunc import toy_potentials
from gala.logging import logger
from gala.potential import (
    Hamiltonian,
    HarmonicOscillatorPotential,
    IsochronePotential,
)
from gala.tests.optional_deps import HAS_TWOBODY
from gala.units import galactic
from gala.util import assert_angles_allclose


class TestIsochrone:
    def setup_method(self):
        logger.info("======== Isochrone ========")
        N = 100
        rng = np.random.default_rng(42)
        x = rng.uniform(-10.0, 10.0, size=(3, N))
        v = rng.uniform(-1.0, 1.0, size=(3, N)) / 33.0
        w0 = np.vstack((x, v))

        self.potential = IsochronePotential(units=galactic, m=1.0e11, b=5.0)
        H = Hamiltonian(self.potential)
        self.w = H.integrate_orbit(w0, dt=0.1, n_steps=10000)
        self.w = self.w[::10]

    def test_single(self):
        n = 13  # MAGIC NUMBER to pick one orbit

        # First, check that value of the actions are stable
        actions, angles, freqs = isochrone_xv_to_aa(self.w[:, n], self.potential)
        for i in range(3):
            assert u.allclose(actions[i, 1:], actions[i, 0], rtol=1e-5)

        for slice_ in [slice(None), 0]:
            actions, angles, _freqs = isochrone_xv_to_aa(
                self.w[slice_, n], self.potential
            )

            # Compare to genfunc
            x = self.w.xyz[:, slice_, n]
            v = self.w.v_xyz[:, slice_, n]
            m = self.potential.parameters["m"].value / 1e11
            b = self.potential.parameters["b"].value

            if x.ndim > 1:
                s_w = np.vstack((x.to_value(u.kpc), v.to_value(u.km / u.s)))

                aa = np.array(
                    [
                        toy_potentials.angact_iso(s_w[:, i].T, params=(m, b))
                        for i in range(s_w.shape[1])
                    ]
                )
                s_actions = aa[:, :3] * u.km / u.s * u.kpc
                s_angles = aa[:, 3:] * u.rad

            else:
                s_w = np.concatenate((x.to_value(u.kpc), v.to_value(u.km / u.s)))

                aa = toy_potentials.angact_iso(s_w.T, params=(m, b))
                s_actions = aa[:3] * u.km / u.s * u.kpc
                s_angles = aa[3:] * u.rad

            assert u.allclose(actions, s_actions.T, rtol=1e-8)
            assert_angles_allclose(angles, s_angles.T, rtol=1e-8)

            # Test round-tripping
            if HAS_TWOBODY:
                w_rt = isochrone_aa_to_xv(actions, angles, self.potential)

                assert u.allclose(x, w_rt.xyz, atol=1e-10 * u.kpc)
                assert u.allclose(v, w_rt.v_xyz, atol=1e-10 * u.km / u.s)

    def test_many(self):
        actions, angles, _freqs = isochrone_xv_to_aa(self.w, self.potential)

        # Compare first element of orbit to genfunc, for speed
        x = self.w.xyz
        v = self.w.v_xyz
        m = self.potential.parameters["m"].value / 1e11
        b = self.potential.parameters["b"].value

        s_w = np.vstack((x[:, 0].to_value(u.kpc), v[:, 0].to_value(u.km / u.s)))

        aa = np.array(
            [
                toy_potentials.angact_iso(s_w[:, i].T, params=(m, b))
                for i in range(s_w.shape[1])
            ]
        )
        s_actions = aa[:, :3] * u.km / u.s * u.kpc
        s_angles = aa[:, 3:] * u.rad

        assert u.allclose(actions[:, 0], s_actions.T, rtol=1e-8)
        assert_angles_allclose(angles[:, 0], s_angles.T, rtol=1e-8)

        # Test round-tripping
        if HAS_TWOBODY:
            # Check round-tripping for full orbits:
            w_rt = isochrone_aa_to_xv(actions, angles, self.potential)

            assert u.allclose(x, w_rt.xyz, atol=1e-10 * u.kpc)
            assert u.allclose(v, w_rt.v_xyz, atol=1e-10 * u.km / u.s)


class TestHarmonicOscillator:
    def setup_method(self):
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
            logger.debug(f"Orbit {n}")

            actions, angles, _freq = harmonic_oscillator_xv_to_aa(
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
