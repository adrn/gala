# Third party
import astropy.units as u
import numpy as np
import pytest

import gala.potential as gp
from gala.integrate import DOPRI853Integrator, LeapfrogIntegrator
from gala.potential.common import PotentialParameter
from gala.units import UnitSystem, galactic, solarsystem


class CompositeHelper:
    rotation = False

    def setup_method(self):
        self.units = solarsystem
        self.p1 = gp.KeplerPotential(m=1.0 * u.Msun, units=self.units)
        self.p2 = gp.HernquistPotential(m=0.5 * u.Msun, c=0.1 * u.au, units=self.units)

    def test_shit(self):
        potential = self.Cls(one=self.p1, two=self.p2)

        q = np.ascontiguousarray(np.array([[1.1, 0, 0]]).T)
        print("val", potential.energy(q))

        q = np.ascontiguousarray(np.array([[1.1, 0, 0]]).T)
        print("grad", potential.gradient(q))

    def test_composite_create(self):
        potential = self.Cls()

        # Add a point mass with same unit system
        potential["one"] = gp.KeplerPotential(units=self.units, m=1.0)

        with pytest.raises(TypeError):
            potential["two"] = "derp"

        assert "one" in potential.parameters
        assert "m" in potential.parameters["one"]
        with pytest.raises(TypeError):
            potential.parameters["m"] = "derp"

    def test_plot_composite(self):
        # TODO: do image comparison or something to compare?

        potential = self.Cls()

        # Add a kepler potential and a harmonic oscillator
        potential["one"] = self.p1
        potential["two"] = self.p2

        grid = np.linspace(-5.0, 5)
        potential.plot_contours(grid=(grid, 0.0, 0.0))
        # fig.savefig(os.path.join(plot_path, "composite_kepler_sho_1d.png"))

        potential.plot_contours(grid=(grid, grid, 0.0))
        # fig.savefig(os.path.join(plot_path, "composite_kepler_sho_2d.png"))

    def test_integrate(self):
        potential = self.Cls()
        potential["one"] = self.p1
        potential["two"] = self.p2

        for Integrator in [DOPRI853Integrator, LeapfrogIntegrator]:
            kw = {}
            if Integrator == DOPRI853Integrator:
                kw = {"atol": 1e-14, "rtol": 1e-14}

            H = gp.Hamiltonian(potential)
            w_cy = H.integrate_orbit(
                [1.0, 0, 0, 0, 2 * np.pi, 0],
                dt=0.01,
                n_steps=1000,
                Integrator=Integrator,
                cython_if_possible=True,
                Integrator_kwargs=kw,
            )
            w_py = H.integrate_orbit(
                [1.0, 0, 0, 0, 2 * np.pi, 0],
                dt=0.01,
                n_steps=1000,
                Integrator=Integrator,
                cython_if_possible=False,
            )

            dx = w_cy.xyz.value - w_py.xyz.value
            assert np.allclose(w_cy.xyz.value, w_py.xyz.value)
            assert np.allclose(w_cy.v_xyz.value, w_py.v_xyz.value)


# ------------------------------------------------------------------------


class TestComposite(CompositeHelper):
    Cls = gp.CompositePotential


class TestCComposite(CompositeHelper):
    Cls = gp.CCompositePotential


def test_failures():
    p = gp.CCompositePotential()
    p["derp"] = gp.KeplerPotential(m=1.0 * u.Msun, units=solarsystem)
    with pytest.raises(ValueError):
        p["jnsdfn"] = gp.HenonHeilesPotential(units=solarsystem)


def test_lock():
    p = gp.CompositePotential()
    p["derp"] = gp.KeplerPotential(m=1.0 * u.Msun, units=solarsystem)
    p.lock = True
    with pytest.raises(ValueError):  # try adding potential after lock
        p["herp"] = gp.KeplerPotential(m=2.0 * u.Msun, units=solarsystem)

    p = gp.CCompositePotential()
    p["derp"] = gp.KeplerPotential(m=1.0 * u.Msun, units=solarsystem)
    p.lock = True
    with pytest.raises(ValueError):  # try adding potential after lock
        p["herp"] = gp.KeplerPotential(m=2.0 * u.Msun, units=solarsystem)


class MyPotential(gp.PotentialBase):
    m = PotentialParameter("m", physical_type="mass")
    x0 = PotentialParameter("x0", physical_type="length")

    def _energy(self, x, t):
        m = self.parameters["m"]
        x0 = self.parameters["x0"]
        r = np.sqrt(np.sum((x - x0[None]) ** 2, axis=1))
        return -m / r

    def _gradient(self, x, t):
        m = self.parameters["m"]
        x0 = self.parameters["x0"]
        r = np.sqrt(np.sum((x - x0[None]) ** 2, axis=1))
        return m * (x - x0[None]) / r**3


def test_add():
    """Test adding potentials to get a composite"""
    p1 = gp.KeplerPotential(units=galactic, m=1 * u.Msun)
    p2 = gp.HernquistPotential(units=galactic, m=1.0e11, c=0.26)

    comp1 = gp.CompositePotential()
    comp1["0"] = p1
    comp1["1"] = p2

    py_p1 = MyPotential(m=1.0, x0=[1.0, 0.0, 0.0], units=galactic)
    py_p2 = MyPotential(m=4.0, x0=[-1.0, 0.0, 0.0], units=galactic)

    # python + python
    new_p = py_p1 + py_p2
    assert isinstance(new_p, gp.CompositePotential)
    assert not isinstance(new_p, gp.CCompositePotential)
    assert len(new_p.keys()) == 2

    # python + python + python
    new_p = py_p1 + py_p2 + py_p2
    assert isinstance(new_p, gp.CompositePotential)
    assert len(new_p.keys()) == 3

    # cython + cython
    new_p = p1 + p2
    assert isinstance(new_p, gp.CCompositePotential)
    assert len(new_p.keys()) == 2

    # cython + python
    new_p = py_p1 + p2
    assert isinstance(new_p, gp.CompositePotential)
    assert not isinstance(new_p, gp.CCompositePotential)
    assert len(new_p.keys()) == 2

    # cython + cython + python
    new_p = p1 + p2 + py_p1
    assert isinstance(new_p, gp.CompositePotential)
    assert not isinstance(new_p, gp.CCompositePotential)
    assert len(new_p.keys()) == 3


def test_no_max_n_components():
    units = UnitSystem(u.pc, u.Myr, u.radian, u.Msun)

    pots = {}
    ms = np.linspace(10, 100, 1024)
    q0s = np.zeros((3, ms.shape[0])) * u.kpc
    q0s[0] = np.linspace(-10, 10, ms.shape[0]) * u.pc
    for i in range(ms.shape[0]):
        pots[f"yo{i}"] = gp.HernquistPotential(
            units=units, m=ms[i] * u.Msun, c=0.1 * u.pc, origin=q0s[:, i]
        )
    comp = gp.CompositePotential(**pots)
