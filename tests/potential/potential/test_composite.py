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

    # C composites only accept C potentials
    ho = gp.HarmonicOscillatorPotential(omega=[1.0, 1.0, 1.0] / u.yr, units=solarsystem)
    with pytest.raises(TypeError):
        p["ho"] = ho
    assert list(p.keys()) == ["derp"]


@pytest.mark.parametrize("Cls", [gp.CompositePotential, gp.CCompositePotential])
def test_lock(Cls):
    p = Cls()
    p["derp"] = gp.KeplerPotential(m=1.0 * u.Msun, units=solarsystem)
    p.lock = True
    with pytest.raises(ValueError):  # try adding potential after lock
        p["herp"] = gp.KeplerPotential(m=2.0 * u.Msun, units=solarsystem)

    # try removing potentials after lock
    with pytest.raises(ValueError):
        del p["derp"]
    with pytest.raises(ValueError):
        p.pop("derp")
    with pytest.raises(ValueError):
        p.popitem()
    with pytest.raises(ValueError):
        p.clear()
    assert list(p.keys()) == ["derp"]


def _halo_disk(Cls):
    pot = Cls()
    pot["halo"] = gp.HernquistPotential(m=1e11, c=5.0, units=galactic)
    pot["disk"] = gp.MiyamotoNagaiPotential(m=1e10, a=3.0, b=0.3, units=galactic)
    return pot


@pytest.mark.parametrize("Cls", [gp.CompositePotential, gp.CCompositePotential])
@pytest.mark.parametrize("how", ["del", "pop", "popitem"])
def test_remove_component(Cls, how):
    pot = _halo_disk(Cls)
    if how == "del":
        del pot["disk"]
    elif how == "pop":
        pot.pop("disk", default=None)
    elif how == "popitem":
        pot.popitem()

    expected = Cls(halo=pot["halo"])
    assert type(pot._symmetry) is type(expected._symmetry)

    # The C potential's mass_enclosed uses the C instance
    q = [8.0, 1.0, 0.5] * u.kpc
    assert u.allclose(pot.mass_enclosed(q), expected.mass_enclosed(q))


@pytest.mark.parametrize("Cls", [gp.CompositePotential, gp.CCompositePotential])
def test_clear(Cls):
    pot = _halo_disk(Cls)
    pot.clear()
    assert len(pot) == 0
    assert pot._symmetry is None
    assert not hasattr(pot, "c_instance")


@pytest.mark.parametrize("Cls", [gp.CompositePotential, gp.CCompositePotential])
def test_hessian_transformed_component(Cls):
    disk = gp.MiyamotoNagaiPotential(m=1e10, a=3.0, b=0.3, units=galactic)
    halo = gp.HernquistPotential(m=1e11, c=2.0, units=galactic)
    x0 = np.array([1.0, 2.0, 3.0])
    x1 = np.array([-1.0, 0.5, 0.0])
    q = np.array([8.0, 1.0, 0.5])

    # A shifted component's hessian needs no correction
    pot = Cls(disk=disk.replicate(origin=x0), halo=halo.replicate(origin=x1))
    expected = disk.hessian(q - x0) + halo.hessian(q - x1)
    assert u.allclose(pot.hessian(q), expected)
    if Cls is gp.CCompositePotential:
        hess = pot.c_instance.hessian(np.ascontiguousarray(q[None]), t=np.array([0.0]))
        assert np.allclose(np.moveaxis(hess, 0, -1), expected.value)

    # A rotated component's hessian is not supported
    c, s = np.cos(0.5), np.sin(0.5)
    R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    pot["tilted"] = disk.replicate(R=R)
    with pytest.raises(NotImplementedError):
        pot.hessian(q)


class MyPotential(gp.PotentialBase):
    m = PotentialParameter("m", physical_type="mass")
    x0 = PotentialParameter("x0", physical_type="length", ndim=1)

    def _energy(self, x, t):
        m = self.parameters["m"]
        x0 = self.parameters["x0"]
        r = np.sqrt(np.sum((x - x0[None]) ** 2, axis=1))
        return -m / r

    def _gradient(self, x, t):
        m = self.parameters["m"]
        x0 = self.parameters["x0"]
        x0 = np.atleast_2d(x0).T
        r = np.sqrt(np.sum((x - x0) ** 2, axis=0))
        return m * (x - x0) / r**3


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
