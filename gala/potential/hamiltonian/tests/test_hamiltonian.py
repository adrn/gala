# coding: utf-8

# Third-party
import astropy.units as u
import pytest

# Project
from .helpers import _TestBase
from .. import Hamiltonian
from ...potential.builtin import KeplerPotential
from ...frame.builtin import StaticFrame, ConstantRotatingFrame
from ....units import solarsystem, galactic, dimensionless

def test_init():
    p = KeplerPotential(m=1.)
    f = StaticFrame()
    H = Hamiltonian(potential=p, frame=f)

    p = KeplerPotential(m=1., units=solarsystem)
    f = StaticFrame(units=solarsystem)
    H = Hamiltonian(potential=p, frame=f)
    H = Hamiltonian(potential=p)

    p = KeplerPotential(m=1.)
    f = StaticFrame(galactic)
    with pytest.raises(ValueError):
        H = Hamiltonian(potential=p, frame=f)

    p = KeplerPotential(m=1., units=solarsystem)
    f = StaticFrame()
    with pytest.raises(ValueError):
        H = Hamiltonian(potential=p, frame=f)

    p = KeplerPotential(m=1., units=solarsystem)
    f = ConstantRotatingFrame(Omega=1./u.yr, units=solarsystem)
    with pytest.raises(ValueError):
        H = Hamiltonian(potential=p, frame=f)
