# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import time

# Third-party
import astropy.units as u
import numpy as np
import pytest
from scipy.misc import derivative
from six.moves import cPickle as pickle

# Project
from ..io import load
from ...units import UnitSystem

def partial_derivative(func, point, ix=0, **kwargs):
    xyz = np.array(point)
    def wraps(a):
        xyz[ix] = a
        return func(xyz)
    return derivative(wraps, point[ix], **kwargs)

class PotentialTestBase(object):

    def setup(self, potential, name=None):
        self.p = potential
        if name is None:
            self.name = potential.__class__.__name__
        else:
            self.name = name
        print("Testing potential: {}".format(self.name))

    def test_unitsystem(self):
        assert isinstance(self.potential.units, UnitSystem)

    def test_value(self):
        pass

    def test_gradient(self):
        pass

    @pytest.mark.skipif(True)
    def test_hessian(self):
        pass

    def test_mass_enclosed(self):
        pass

    def test_repr(self):
        pass

    def test_energy(self):
        pass

    def test_plot(self):
        # TODO
        pass

    def test_save_load(self, tmpdir):
        """
        Test writing to a YAML file, and reading back in
        """
        fn = str(tmpdir.join("{}.yml".format(self.name)))
        self.potential.save(fn)
        p = load(fn)
        p.value([100,0,0.])

    def test_numerical_gradient_vs_gradient(self):
        """
        Check that the value of the implemented gradient function is close to a
        numerically estimated value. This is to check the coded-up version.
        """

        dx = 1E-6
        max_x = np.sqrt(np.sum([x**2 for x in self.w0[:3]]))

        grid = np.linspace(-max_x,max_x,8)
        grid = grid[grid != 0.]
        xyz = np.vstack(map(np.ravel, np.meshgrid(grid,grid,grid))).T

        num_grad = np.zeros_like(xyz)
        for i in range(xyz.shape[0]):
            num_grad[i] = np.array([partial_derivative(self.potential.value, xyz[i], ix=ix, n=1, dx=dx, order=5) for ix in range(3)]).T
        grad = self.potential.gradient(xyz)
        np.testing.assert_allclose(num_grad, grad, rtol=1E-5)

    def test_orbit_integration(self):
        """
        Make we can integrate an orbit in this potential
        """
        w0 = self.w0
        t1 = time.time()
        t,w = self.potential.integrate_orbit(w0, dt=1., nsteps=10000)
        print("Integration time (10000 steps): {}".format(time.time() - t1))

    def test_pickle(self, tmpdir):
        fn = str(tmpdir.join("{}.pickle".format(self.name)))
        with open(fn, "w") as f:
            pickle.dump(self.potential, f)

        with open(fn) as f:
            p = pickle.load(f)

        p.value([100,0,0.])
