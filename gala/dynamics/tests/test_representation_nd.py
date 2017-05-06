# coding: utf-8

from __future__ import division, print_function

# Third-party
import astropy.units as u
from astropy.coordinates import Galactic, CartesianRepresentation
from astropy.tests.helper import quantity_allclose
import numpy as np
import pytest

# Project
from ..core import PhaseSpacePosition
from ...potential import Hamiltonian, HernquistPotential
from ...potential.frame import StaticFrame, ConstantRotatingFrame
from ...units import galactic, solarsystem
from ..representation_nd import (NDCartesianRepresentation,
                                 NDCartesianDifferential)

def test_init_repr():

    # Passing in x1, x2
    rep = NDCartesianRepresentation([1., 1.])
    assert rep.xyz.shape == (2,)

    # Passing in x1, x2
    rep = NDCartesianRepresentation(np.random.random(size=(2, 8)))
    assert rep.xyz.shape == (2,8)
    rep[:1]

    for n in range(1, 6+1):
        print('N: '+str(n))

        xs = np.random.uniform(size=(n, 16)) * u.one
        rep = NDCartesianRepresentation(xs)
        for i in range(1, n+1):
            assert hasattr(rep, 'x'+str(i))

        xs2 = rep.xyz
        assert quantity_allclose(xs, xs2)

        rep2 = rep[:8]

        assert rep.shape == (16,)
        assert rep2.shape == (8,)

def test_init_diff():

    # Passing in x1, x2
    rep = NDCartesianDifferential([1., 1.])
    assert rep.d_xyz.shape == (2,)
    with pytest.raises(TypeError):
        rep[:1]

    # Passing in x1, x2
    rep = NDCartesianDifferential(np.random.random(size=(2, 8)))
    assert rep.d_xyz.shape == (2,8)
    rep[:1]

    for n in range(1, 6+1):
        print('N: '+str(n))

        xs = np.random.uniform(size=(n, 16)) * u.one
        rep = NDCartesianDifferential(xs)
        for i in range(1, n+1):
            assert hasattr(rep, 'd_x'+str(i))

        xs2 = rep.d_xyz
        assert quantity_allclose(xs, xs2)

        rep2 = rep[:8]

        assert rep.shape == (16,)
        assert rep2.shape == (8,)
