"""
    Test the Cython integrators.
"""

# Standard library
import time

# Third-party
import numpy as np
import pytest

# Project
from ..pyintegrators.leapfrog import LeapfrogIntegrator
from ..cyintegrators.leapfrog import leapfrog_integrate_hamiltonian
from ..pyintegrators.dopri853 import DOPRI853Integrator
from ..cyintegrators.dop853 import dop853_integrate_hamiltonian
from ..pyintegrators.ruth4 import Ruth4Integrator
from ..cyintegrators.ruth4 import ruth4_integrate_hamiltonian
from ...potential import Hamiltonian, HernquistPotential
from ...units import galactic

integrator_list = [LeapfrogIntegrator, DOPRI853Integrator, Ruth4Integrator]
func_list = [
    leapfrog_integrate_hamiltonian,
    dop853_integrate_hamiltonian,
    ruth4_integrate_hamiltonian,
]

_list = []
for dt in [2, -2]:
    _list.extend([(x, y, dt) for x, y in zip(integrator_list, func_list)])


@pytest.mark.parametrize(
    ["Integrator", "integrate_func", "dt"],
    _list
)
def test_compare_to_py(Integrator, integrate_func, dt):
    p = HernquistPotential(m=1e11, c=0.5, units=galactic)
    H = Hamiltonian(potential=p)

    def F(t, w):
        w_T = np.ascontiguousarray(w.T)
        return H._gradient(w_T, np.array([0.0])).T

    cy_w0 = np.array(
        [
            [0.0, 10.0, 0.0, 0.2, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0, 0.2, 0.0],
            [0.0, 10.0, 0.0, 0.0, 0.0, 0.2],
        ]
    )
    py_w0 = np.ascontiguousarray(cy_w0.T)

    n_steps = 1024
    t = np.linspace(0, dt * n_steps, n_steps + 1)

    cy_t, cy_w = integrate_func(H, cy_w0, t)
    cy_w = np.rollaxis(cy_w, -1)

    integrator = Integrator(F)
    orbit = integrator.run(py_w0, dt=dt, n_steps=n_steps)

    py_t = orbit.t.value
    py_w = orbit.w()

    assert py_w.shape == cy_w.shape
    assert np.allclose(cy_w[:, -1], py_w[:, -1])
    assert np.allclose(cy_t, py_t)


# TODO: move this to only run if a flag like --remote-data is passed, like
# --speed-scaling or something?
@pytest.mark.skipif(True, reason="Slow test - mainly for plotting locally")
@pytest.mark.parametrize(
    ["Integrator", "integrate_func"],
    zip(integrator_list, func_list)
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

    for c, nparticles in zip(colors, [1, 100, 1000]):
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
            orbit = integrator.run(py_w0, dt=dt, n_steps=n_steps)
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
