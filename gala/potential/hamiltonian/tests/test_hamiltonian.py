import pickle

import astropy.units as u
import pytest

from ....units import galactic, solarsystem
from ...frame.builtin import ConstantRotatingFrame, StaticFrame
from ...potential.builtin import KeplerPotential
from .. import Hamiltonian


def test_init():
    p = KeplerPotential(m=1.0)
    f = StaticFrame()
    H = Hamiltonian(potential=p, frame=f)
    H2 = Hamiltonian(H)
    assert H2.potential is H.potential

    p = KeplerPotential(m=1.0, units=solarsystem)
    f = StaticFrame(units=solarsystem)
    H = Hamiltonian(potential=p, frame=f)
    H = Hamiltonian(potential=p)

    p = KeplerPotential(m=1.0)
    f = StaticFrame(galactic)
    with pytest.raises(ValueError):
        H = Hamiltonian(potential=p, frame=f)

    p = KeplerPotential(m=1.0, units=solarsystem)
    f = StaticFrame()
    with pytest.raises(ValueError):
        H = Hamiltonian(potential=p, frame=f)

    p = KeplerPotential(m=1.0, units=solarsystem)
    f = ConstantRotatingFrame(Omega=1.0 / u.yr, units=solarsystem)
    with pytest.raises(ValueError):
        H = Hamiltonian(potential=p, frame=f)


def test_pickle(tmpdir):
    filename = tmpdir / "hamil.pkl"

    p = KeplerPotential(m=1.0, units=solarsystem)

    for fr in [
        StaticFrame(units=solarsystem),
        ConstantRotatingFrame(Omega=[0, 0, 1] / u.yr, units=solarsystem),
    ]:
        H = Hamiltonian(potential=p, frame=fr)

        with open(filename, "wb") as f:
            pickle.dump(H, f)

        with open(filename, "rb") as f:
            H2 = pickle.load(f)
