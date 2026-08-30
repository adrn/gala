"""
Test the Cython integrators.
"""

import time
from itertools import product

import numpy as np
import pytest
from gala.integrate.cyintegrators.dop853 import dop853_integrate_hamiltonian
from gala.integrate.cyintegrators.leapfrog import leapfrog_integrate_hamiltonian
from gala.integrate.cyintegrators.ruth4 import ruth4_integrate_hamiltonian

from gala.integrate.pyintegrators.dopri853 import DOPRI853Integrator
from gala.integrate.pyintegrators.leapfrog import LeapfrogIntegrator
from gala.integrate.pyintegrators.ruth4 import Ruth4Integrator
from gala.potential import Hamiltonian, HernquistPotential
from gala.units import galactic

integrator_list = [LeapfrogIntegrator, DOPRI853Integrator, Ruth4Integrator]
func_list = [
    leapfrog_integrate_hamiltonian,
    dop853_integrate_hamiltonian,
    ruth4_integrate_hamiltonian,
]

_list = []
for dt in [2, -2]:
    _list.extend([(x, y, dt) for x, y in zip(integrator_list, func_list)])


@pytest.mark.parametrize(("Integrator", "integrate_func", "dt"), _list)
def test_compare_to_py(Integrator, integrate_func, dt):
    p = HernquistPotential(m=1e11, c=0.5, units=galactic)
    H = Hamiltonian(potential=p)

    def F(t, w):
        w = np.ascontiguousarray(w)
        return H._gradient(w, np.array([0.0]))

    cy_w0 = np.array(
        [
            [0.0, 10.0, 0.0, 0.2, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0, 0.2, 0.0],
            [0.0, 10.0, 0.0, 0.0, 0.0, 0.2],
        ]
    )
    cy_w0 = np.ascontiguousarray(cy_w0.T)
    py_w0 = cy_w0.copy()

    n_steps = 1024
    t = np.linspace(0, dt * n_steps, n_steps + 1)

    cy_t, cy_w = integrate_func(H, cy_w0, t)

    integrator = Integrator(F)
    orbit = integrator(py_w0, dt=dt, n_steps=n_steps)

    py_t = orbit.t.value
    py_w = orbit.w()  # (ndim, ntimes, n)

    assert py_w.shape == cy_w.shape
    assert np.allclose(cy_w[:, -1], py_w[:, -1])
    assert np.allclose(cy_t, py_t)


@pytest.mark.parametrize(("integrate_func", "dt"), product(func_list, [-2.0, 2]))
def test_save_all(integrate_func, dt):
    p = HernquistPotential(m=1e11, c=0.5, units=galactic)
    H = Hamiltonian(potential=p)

    w0 = np.array(
        [
            [0.0, 10.0, 0.0, 0.2, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0, 0.2, 0.0],
            [0.0, 10.0, 0.0, 0.0, 0.0, 0.2],
        ]
    )
    w0 = np.ascontiguousarray(w0.T)

    # 1024 steps
    t = np.linspace(0, dt * 1024, 1024 + 1)

    t_all, w_all = integrate_func(H, w0, t)
    t_f, w_f = integrate_func(H, w0, t, save_all=False)

    assert t_all[-1] == t_f[0]
    assert np.allclose(w_all[:, -1], w_f)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize(
    "times",
    [
        pytest.param(np.array([1.0, 1.0]), id="zero-duration"),
        pytest.param(np.array([1e-3, 1e-3 + 1e-18]), id="tiny-final-step"),
        pytest.param(
            np.array(
                [
                    0.0,
                    0.0022884973945761,
                    0.0973861319484548,
                    1.653381907704304,
                    1.729685002780781,
                ]
            ),
            id="irregular-dense-output",
        ),
    ],
)
def test_dop853_time_array_roundoff(direction, times):
    p = HernquistPotential(m=1e11, c=0.5, units=galactic)
    H = Hamiltonian(potential=p)
    w0 = np.ascontiguousarray(np.array([[0.0, 10.0, 0.0, 0.2, 0.0, 0.0]]).T)
    t = direction * times

    t_all, w_all = dop853_integrate_hamiltonian(H, w0, t)
    t_f, w_f = dop853_integrate_hamiltonian(H, w0, t, save_all=False)

    np.testing.assert_array_equal(t_all, t)
    np.testing.assert_array_equal(t_f, t[-1:])
    np.testing.assert_allclose(w_all[:, 0], w0)
    assert np.isfinite(w_all).all()
    np.testing.assert_allclose(w_all[:, -1], w_f)

    if np.all(times == times[0]):
        expected = np.repeat(w0[:, None], len(t), axis=1)
        np.testing.assert_array_equal(w_all, expected)
        np.testing.assert_array_equal(w_f, w0)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_dop853_rejected_rounded_endpoint(direction):
    p = HernquistPotential(m=1e11, c=0.5, units=galactic)
    H = Hamiltonian(potential=p)
    w0 = np.ascontiguousarray(np.array([[0.0, 10.0, 0.0, 0.2, 0.0, 0.0]]).T)
    start = direction * 1e17
    end = np.nextafter(start, direction * np.inf)

    with pytest.raises(RuntimeError, match=r"Integration failed with code -3"):
        dop853_integrate_hamiltonian(
            H,
            w0,
            np.array([start, end]),
            atol=1e-10,
            rtol=1e-10,
            nmax=1,
        )


# TODO: move this to only run if a flag like --remote-data is passed, like
# --speed-scaling or something?
@pytest.mark.skipif(True, reason="Slow test - mainly for plotting locally")
@pytest.mark.parametrize(
    ("Integrator", "integrate_func"), zip(integrator_list, func_list)
)
def test_scaling(tmpdir, Integrator, integrate_func):
    p = HernquistPotential(m=1e11, c=0.5, units=galactic)

    def F(t, w):
        dq = w[3:]
        dp = -p._gradient(w[:3], t=np.array([0.0]))
        return np.vstack((dq, dp))

    step_bins = np.logspace(2, np.log10(25000), 7)
    colors = ["k", "b", "r"]
    dt = 1.0

    for _c, nparticles in zip(colors, [1, 100, 1000]):
        cy_w0 = np.array([[0.0, 10.0, 0.0, 0.2, 0.0, 0.0]] * nparticles)
        py_w0 = np.ascontiguousarray(cy_w0.T)

        x = []
        cy_times = []
        py_times = []
        for n_steps in step_bins:
            print(nparticles, n_steps)
            t = np.linspace(0, dt * n_steps, n_steps + 1)
            x.append(n_steps)

            # time the Cython integration
            t0 = time.time()
            integrate_func(p.c_instance, cy_w0, t)
            cy_times.append(time.time() - t0)

            # time the Python integration
            t0 = time.time()
            integrator = Integrator(F)
            orbit = integrator(py_w0, dt=dt, n_steps=n_steps)
            py_times.append(time.time() - t0)

    #     pl.loglog(x, cy_times, linestyle='-', lw=2., c=c, marker='',
    #               label="cy: {} orbits".format(nparticles))
    #     pl.loglog(x, py_times, linestyle='--', lw=2., c=c, marker='',
    #               label="py: {} orbits".format(nparticles))

    # pl.title(Integrator.__name__)
    # pl.legend(loc='upper left')
    # pl.xlim(90, 30000)
    # pl.xlabel("N steps")
    # pl.tight_layout()
    # # pl.show()
    # pl.savefig(os.path.join(tmpdir, "integrate-scaling.png"), dpi=300)
