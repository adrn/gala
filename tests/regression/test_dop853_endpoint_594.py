"""Requested endpoints must be included despite roundoff in the last step."""

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

import gala.dynamics as gd
import gala.potential as gp

TIMES = np.array(
    [0.0, 0.0022884973945761, 0.0973861319484548, 1.653381907704304, 1.729685002780781]
)


@pytest.mark.parametrize("origin", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("direction", [-1, 1])
def test_milky_way_dense_endpoint(origin, direction):
    potential = gp.MilkyWayPotential(version="latest")
    w0 = gd.PhaseSpacePosition(
        pos=np.array([-8.122, 0.0, 0.0208]) * u.kpc,
        vel=np.array([12.9, 245.6, 7.78]) * u.km / u.s,
    )
    times = (origin + direction * TIMES) * u.Myr
    dense = potential.integrate_orbit(w0, t=times, Integrator="dop853")
    terminal = potential.integrate_orbit(
        w0, t=times, Integrator="dop853", save_all=False
    )
    python = potential.integrate_orbit(
        w0, t=times, Integrator="dop853", cython_if_possible=False
    )

    assert_allclose(
        dense.w(potential.units)[:, -1],
        terminal.w(potential.units)[:, 0],
        rtol=1e-12,
        atol=1e-12,
    )
    assert_allclose(
        dense.w(potential.units),
        python.w(potential.units),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("origin", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("direction", [-1, 1])
def test_kepler_dense_output_against_circle(origin, direction):
    potential = gp.KeplerPotential(m=1.0)
    elapsed = direction * TIMES
    orbit = potential.integrate_orbit(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        t=origin + elapsed,
        Integrator="dop853",
        Integrator_kwargs={"atol": 1e-12, "rtol": 1e-12},
    )
    expected = np.array(
        [
            np.cos(elapsed),
            np.sin(elapsed),
            np.zeros_like(elapsed),
            -np.sin(elapsed),
            np.cos(elapsed),
            np.zeros_like(elapsed),
        ]
    )
    assert_allclose(orbit.w(), expected, rtol=1e-10, atol=1e-10)
