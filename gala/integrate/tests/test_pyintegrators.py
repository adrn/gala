"""
Test the integrators.
"""

import os

import numpy as np
import pytest
from astropy.utils.exceptions import AstropyDeprecationWarning

from gala.tests.optional_deps import HAS_TQDM

from .. import (
    DOPRI853Integrator,
    LeapfrogIntegrator,
    RK5Integrator,
    Ruth4Integrator,
)

# Integrators to test
integrator_list = [
    RK5Integrator,
    DOPRI853Integrator,
    LeapfrogIntegrator,
    Ruth4Integrator,
]


# Gradient functions:
def sho_F(t, w, T):
    """Simple harmonic oscillator"""
    q, p = w
    wdot = np.zeros_like(w)
    wdot[0] = p
    wdot[1] = -((2 * np.pi / T) ** 2) * q
    return wdot


def forced_sho_F(t, w, A, omega_d):
    q, p = w
    wdot = np.zeros_like(w)
    wdot[0] = p
    wdot[1] = -np.sin(q) + A * np.cos(omega_d * t)
    return wdot


def lorenz_F(t, w, sigma, rho, beta):
    x, y, z, *_ = w
    wdot = np.zeros_like(w)
    wdot[0] = sigma * (y - x)
    wdot[1] = x * (rho - z) - y
    wdot[2] = x * y - beta * z
    return wdot


def ptmass_F(t, w):
    x, y, px, py = w
    a = -1.0 / (x * x + y * y) ** 1.5

    wdot = np.zeros_like(w)
    wdot[0] = px
    wdot[1] = py
    wdot[2] = x * a
    wdot[3] = y * a
    return wdot


@pytest.mark.parametrize("Integrator", integrator_list)
def test_sho_forward_backward(Integrator):
    integrator = Integrator(sho_F, func_args=(1.0,))

    dt = 1e-4
    n_steps = 10_000

    forw = integrator([0.0, 1.0], dt=dt, n_steps=n_steps)
    back = integrator([0.0, 1.0], dt=-dt, n_steps=n_steps)

    assert np.allclose(forw.w()[:, -1], back.w()[:, -1], atol=1e-6)


@pytest.mark.parametrize("Integrator", integrator_list)
def test_deprecated_run_method(Integrator):
    """Test the deprecated run method."""
    integrator = Integrator(sho_F, func_args=(1.0,))

    dt = 1e-4
    n_steps = 10_000

    with pytest.warns(AstropyDeprecationWarning):
        run = integrator.run([0.0, 1.0], dt=dt, n_steps=n_steps)

    call = integrator([0.0, 1.0], dt=dt, n_steps=n_steps)

    assert np.allclose(run.w()[:, -1], call.w()[:, -1], atol=1e-6)


@pytest.mark.parametrize("Integrator", integrator_list)
def test_point_mass(Integrator):
    q0 = np.array([1.0, 0.0])
    p0 = np.array([0.0, 1.0])

    integrator = Integrator(ptmass_F)
    orbit = integrator(np.append(q0, p0), t1=0.0, t2=2 * np.pi, n_steps=1e4)

    assert np.allclose(orbit.w()[:, 0], orbit.w()[:, -1], atol=1e-6)


@pytest.mark.skipif(not HAS_TQDM, reason="requires tqdm to run this test")
@pytest.mark.parametrize("Integrator", integrator_list)
def test_progress(Integrator):
    q0 = np.array([1.0, 0.0])
    p0 = np.array([0.0, 1.0])

    integrator = Integrator(ptmass_F, progress=True)
    _ = integrator(np.append(q0, p0), t1=0.0, t2=2 * np.pi, n_steps=1e2)


@pytest.mark.parametrize("Integrator", integrator_list)
def test_point_mass_multiple(Integrator):
    w0 = np.array([[1.0, 0.0, 0.0, 1.0], [0.8, 0.0, 0.0, 1.1], [2.0, 1.0, -1.0, 1.1]]).T

    integrator = Integrator(ptmass_F)
    _ = integrator(w0, dt=1e-3, n_steps=1e4)


@pytest.mark.parametrize("Integrator", integrator_list)
def test_driven_pendulum(Integrator):
    integrator = Integrator(forced_sho_F, func_args=(0.07, 0.75))
    _ = integrator([3.0, 0.0], dt=1e-2, n_steps=1e4)


@pytest.mark.parametrize("Integrator", integrator_list)
def test_lorenz(Integrator):
    sigma, rho, beta = 10.0, 28.0, 8 / 3.0
    integrator = Integrator(lorenz_F, func_args=(sigma, rho, beta))

    _ = integrator([0.5, 0.5, 0.5, 0, 0, 0], dt=1e-2, n_steps=1e4)


@pytest.mark.parametrize("Integrator", integrator_list)
def test_memmap(tmpdir, Integrator):
    dt = 0.1
    n_steps = 1000
    nw0 = 10000

    filename = os.path.join(str(tmpdir), "test_memmap.npy")
    mmap = np.memmap(filename, mode="w+", shape=(2, n_steps + 1, nw0))

    w0 = np.random.uniform(-1, 1, size=(2, nw0))

    integrator = Integrator(sho_F, func_args=(1.0,))

    _ = integrator(w0, dt=dt, n_steps=n_steps, mmap=mmap)


@pytest.mark.parametrize("Integrator", integrator_list)
def test_py_save_all(Integrator):
    integrator_all = Integrator(sho_F, func_args=(1.3,), save_all=True)
    integrator_final = Integrator(sho_F, func_args=(1.3,), save_all=False)

    dt = 1e-4
    n_steps = 10_000

    out_all = integrator_all([0.0, 1.0], dt=dt, n_steps=n_steps)
    out_final = integrator_final([0.0, 1.0], dt=dt, n_steps=n_steps)

    assert np.allclose(out_all.w()[:, -1], out_final.w()[:, 0])
