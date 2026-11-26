"""
Test the Cython integrators.
"""

import time
from itertools import product

import numpy as np
import pytest
from gala._cconfig import GSL_ENABLED
from gala.integrate.cyintegrators.dop853 import dop853_integrate_hamiltonian
from gala.integrate.cyintegrators.leapfrog import leapfrog_integrate_hamiltonian
from gala.integrate.cyintegrators.ruth4 import ruth4_integrate_hamiltonian

from gala.integrate.pyintegrators.dopri853 import DOPRI853Integrator
from gala.integrate.pyintegrators.leapfrog import LeapfrogIntegrator
from gala.integrate.pyintegrators.ruth4 import Ruth4Integrator
from gala.potential import Hamiltonian, HernquistPotential
from gala.units import galactic


@pytest.mark.skipif(not GSL_ENABLED, reason="requires GSL for time interpolation")
@pytest.mark.parametrize("nbody", [False, True])
@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("save_all", [False, True])
def test_ruth4_time_dependent_order(nbody, direction, save_all):
    from scipy.integrate import solve_ivp

    from gala.dynamics import PhaseSpacePosition
    from gala.dynamics.nbody import DirectNBody
    from gala.potential import KeplerPotential, TimeInterpolatedPotential

    potential = TimeInterpolatedPotential(
        potential_cls=KeplerPotential,
        time_knots=[0.0, 10.0],
        m=[1.0, 2.0],
        interpolation_method="linear",
    )
    w0 = np.array([[1, 0, 0, 0, 1, 0], [0, 1.5, 0, -0.8, 0, 0]]).T
    t0 = 1.3 if direction == 1 else 4.3
    t1 = t0 + direction * 3

    def force(t, w):
        q, v = w.reshape(2, 3, 2)
        return np.array([v, -(1 + t / 10) * q / np.linalg.norm(q, axis=0) ** 3]).ravel()

    reference = (
        solve_ivp(force, (t0, t1), w0.ravel(), method="DOP853", rtol=1e-12, atol=1e-13)
        .y[:, -1]
        .reshape(6, 2)
    )
    errors = []
    for n_steps in [60, 120]:
        times = np.linspace(t0, t1, n_steps + 1)
        if nbody:
            system = DirectNBody(
                PhaseSpacePosition.from_w(w0),
                particle_potentials=[None, None],
                external_potential=potential,
                save_all=save_all,
            )
            result = system.integrate_orbit(Integrator=Ruth4Integrator, t=times).w()
        else:
            _, result = ruth4_integrate_hamiltonian(
                Hamiltonian(potential),
                np.ascontiguousarray(w0),
                times,
                save_all=save_all,
            )
        if save_all:
            result = result[:, -1]
        errors.append(np.max(np.abs(result - reference)))

    assert errors[0] / errors[1] > 2**3.5


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
