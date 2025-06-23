"""
Test the core Potential classes
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import G

from gala.potential import CompositePotential, PotentialBase, PotentialParameter
from gala.tests.optional_deps import HAS_MATPLOTLIB
from gala.units import UnitSystem

if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt
else:
    plt = None


units = [u.kpc, u.Myr, u.Msun, u.radian]
usys = UnitSystem(u.au, u.yr, u.Msun, u.radian)
G = G.decompose(units)


def test_new_simple():
    class MyPotential(PotentialBase):
        ndim = 1

        def _energy(self, r, t=0.0):
            return -1 / r

        def _gradient(self, r, t=0.0):
            return r**-2

    p = MyPotential()
    assert p(0.5) == -2.0
    assert p.energy(0.5) == -2.0
    assert p.acceleration(0.5) == -4.0

    p(np.arange(0.5, 11.5, 0.5).reshape(1, -1))
    p.energy(np.arange(0.5, 11.5, 0.5).reshape(1, -1))
    p.acceleration(np.arange(0.5, 11.5, 0.5).reshape(1, -1))


class MyPotential(PotentialBase):
    m = PotentialParameter("m", "mass")
    x0 = PotentialParameter("x0", "length")
    filename = PotentialParameter("filename", None)
    n = PotentialParameter("n", physical_type=None, default=2)

    def _energy(self, x, t):
        m = self.parameters["m"].value
        x0 = self.parameters["x0"].value
        r = np.sqrt(np.sum((x - x0[None]) ** 2, axis=1))
        return -m / r

    def _gradient(self, x, t):
        m = self.parameters["m"].value
        x0 = self.parameters["x0"].value
        r = np.sqrt(np.sum((x - x0[None]) ** 2, axis=1))
        return m * (x - x0[None]) / r**3


def test_init_potential():
    MyPotential(1.5, 1, "blah")
    MyPotential(1.5, x0=1, filename="blah")
    MyPotential(m=1.5, x0=1, filename="blah")
    MyPotential(1.5 * u.Msun, 1 * u.au, "blah", 10, units=usys)
    MyPotential(1.5 * u.Msun, x0=1 * u.au, filename="blah", units=usys)
    MyPotential(m=1.5 * u.Msun, x0=1 * u.au, filename="blah", units=usys)

    pot = MyPotential(m=1.5 * u.Msun, x0=1 * u.au, n=10, filename="blah", units=usys)
    assert pot.parameters["n"] == 10
    assert pot.parameters["filename"] == "blah"


def test_repr():
    p = MyPotential(m=1.0e10 * u.Msun, x0=0.0, units=usys)
    repr_ = repr(p)
    assert repr_.startswith("<MyPotential: m=")
    assert "m=1" in repr_
    assert "x0=0" in repr_
    assert repr_.endswith("(AU,yr,solMass,rad)>")


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib is required")
def test_plot():
    # TODO: test that the plots are correct??
    p = MyPotential(m=1, x0=[1.0, 3.0, 0.0], units=usys)
    f = p.plot_contours(grid=(np.linspace(-10.0, 10.0, 100), 0.0, 0.0), labels=["X"])
    plt.close(f)
    # f.suptitle("slice off from 0., won't have cusp")
    # f.savefig(os.path.join(plot_path, "contour_x.png"))

    f = p.plot_contours(
        grid=(
            np.linspace(-10.0, 10.0, 100),
            np.linspace(-10.0, 10.0, 100),
            0.0,
        ),
        cmap="Blues",
    )
    plt.close(f)
    # f.savefig(os.path.join(plot_path, "contour_xy.png"))

    f = p.plot_contours(
        grid=(
            np.linspace(-10.0, 10.0, 100),
            1.0,
            np.linspace(-10.0, 10.0, 100),
        ),
        cmap="Blues",
        labels=["X", "Z"],
    )
    plt.close(f)
    # f.savefig(os.path.join(plot_path, "contour_xz.png"))


def test_composite():
    p1 = MyPotential(m=1.0, x0=[1.0, 0.0, 0.0], units=usys)
    p2 = MyPotential(m=1.0, x0=[-1.0, 0.0, 0.0], units=usys)

    p = CompositePotential(one=p1, two=p2)
    assert u.allclose(p.energy([0.0, 0.0, 0.0]), -2 * usys["energy"] / usys["mass"])
    assert u.allclose(p.acceleration([0.0, 0.0, 0.0]), 0.0 * usys["acceleration"])

    p1 = MyPotential(m=1.0, x0=[1.0, 0.0, 0.0], units=usys)
    p2 = MyPotential(m=1.0, x0=[-1.0, 0.0, 0.0], units=[u.kpc, u.yr, u.Msun, u.radian])
    with pytest.raises(ValueError):
        p = CompositePotential(one=p1, two=p2)

    p1 = MyPotential(m=1.0, x0=[1.0, 0.0, 0.0], units=usys)
    p2 = MyPotential(m=1.0, x0=[-1.0, 0.0, 0.0], units=usys)
    p = CompositePotential(one=p1, two=p2)
    assert u.au in p.units
    assert u.yr in p.units
    assert u.Msun in p.units


def test_replace_units():
    usys1 = UnitSystem([u.kpc, u.Gyr, u.Msun, u.radian])
    usys2 = UnitSystem([u.pc, u.Myr, u.Msun, u.degree])

    p = MyPotential(m=1.0e10 * u.Msun, x0=0.0, units=usys1)
    assert p.parameters["m"].unit == usys1["mass"]

    p2 = p.replace_units(usys2)
    assert p2.parameters["m"].unit == usys2["mass"]
    assert p.units == usys1
    assert p2.units == usys2


def test_replicate():
    usys = UnitSystem([u.kpc, u.Gyr, u.Msun, u.radian])
    R = np.diag(np.arange(3))
    p1 = MyPotential(m=1.0e10 * u.Msun, x0=0.0, units=usys, R=R)
    p2 = p1.replicate(m=2e10 * u.Msun, R=None)

    assert p2.R is None
    assert np.isclose(p2.parameters["m"].value, 2e10)
    assert np.isclose(p2.parameters["x0"].value, p1.parameters["x0"].value)
    assert p2.parameters["n"] == p1.parameters["n"]
