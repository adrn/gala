# coding: utf-8

from __future__ import division, print_function


# Standard library
import time

# Third-party
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative
from astropy.extern.six.moves import cPickle as pickle
import pytest

# Project
from ..io import load
from ..core import CompositePotential
from ....units import UnitSystem, DimensionlessUnitSystem
from ....dynamics import PhaseSpacePosition
from ....integrate import LeapfrogIntegrator

def partial_derivative(func, point, dim_ix=0, **kwargs):
    xyz = np.array(point, copy=True)
    def wraps(a):
        xyz[dim_ix] = a
        return func(xyz)
    return derivative(wraps, point[dim_ix], **kwargs)

class PotentialTestBase(object):
    name = None
    potential = None # MUST SET THIS
    tol = 1E-5
    show_plots = False

    @classmethod
    def setup_class(cls):
        if cls.name is None:
            cls.name = cls.__name__[4:] # remove Test
        print("Testing potential: {}".format(cls.name))
        cls.w0 = np.array(cls.w0)
        cls.ndim = cls.w0.size // 2

        # TODO: need to test also quantity objects and phasespacepositions!

        # these are arrays we will test the methods on:
        w0_2d = np.repeat(cls.w0[:,None], axis=1, repeats=16)
        w0_3d = np.repeat(w0_2d[...,None], axis=2, repeats=8)
        w0_list = list(cls.w0)
        w0_slice = w0_2d[:,:4]
        cls.w0s = [cls.w0, w0_2d, w0_3d, w0_list, w0_slice]
        cls._grad_return_shapes = [cls.w0[:cls.ndim].shape + (1,),
                                   w0_2d[:cls.ndim].shape,
                                   w0_3d[:cls.ndim].shape,
                                   cls.w0[:cls.ndim].shape + (1,),
                                   w0_slice[:cls.ndim].shape]
        cls._hess_return_shapes = [(cls.ndim,) + cls.w0[:cls.ndim].shape + (1,),
                                   (cls.ndim,) + w0_2d[:cls.ndim].shape,
                                   (cls.ndim,) + w0_3d[:cls.ndim].shape,
                                   (cls.ndim,) + cls.w0[:cls.ndim].shape + (1,),
                                   (cls.ndim,) + w0_slice[:cls.ndim].shape]
        cls._valu_return_shapes = [x[1:] for x in cls._grad_return_shapes]

    def test_unitsystem(self):
        assert isinstance(self.potential.units, UnitSystem)

    def test_energy(self):
        assert self.ndim == self.potential.ndim

        for arr,shp in zip(self.w0s, self._valu_return_shapes):
            v = self.potential.energy(arr[:self.ndim])
            assert v.shape == shp

            g = self.potential.energy(arr[:self.ndim], t=0.1)
            g = self.potential.energy(arr[:self.ndim], t=0.1*self.potential.units['time'])

            t = np.zeros(np.array(arr).shape[1:]) + 0.1
            g = self.potential.energy(arr[:self.ndim], t=t)
            g = self.potential.energy(arr[:self.ndim], t=t*self.potential.units['time'])

    def test_gradient(self):
        for arr,shp in zip(self.w0s, self._grad_return_shapes):
            g = self.potential.gradient(arr[:self.ndim])
            assert g.shape == shp

            g = self.potential.gradient(arr[:self.ndim], t=0.1)
            g = self.potential.gradient(arr[:self.ndim], t=0.1*self.potential.units['time'])

            t = np.zeros(np.array(arr).shape[1:]) + 0.1
            g = self.potential.gradient(arr[:self.ndim], t=t)
            g = self.potential.gradient(arr[:self.ndim], t=t*self.potential.units['time'])

    def test_hessian(self):
        for arr,shp in zip(self.w0s, self._hess_return_shapes):
            g = self.potential.hessian(arr[:self.ndim])
            assert g.shape == shp

            g = self.potential.hessian(arr[:self.ndim], t=0.1)
            g = self.potential.hessian(arr[:self.ndim], t=0.1*self.potential.units['time'])

            t = np.zeros(np.array(arr).shape[1:]) + 0.1
            g = self.potential.hessian(arr[:self.ndim], t=t)
            g = self.potential.hessian(arr[:self.ndim], t=t*self.potential.units['time'])

    def test_mass_enclosed(self):
        for arr,shp in zip(self.w0s, self._valu_return_shapes):
            g = self.potential.mass_enclosed(arr[:self.ndim])
            assert g.shape == shp
            assert np.all(g > 0.)

            g = self.potential.mass_enclosed(arr[:self.ndim], t=0.1)
            g = self.potential.mass_enclosed(arr[:self.ndim], t=0.1*self.potential.units['time'])

            t = np.zeros(np.array(arr).shape[1:]) + 0.1
            g = self.potential.mass_enclosed(arr[:self.ndim], t=t)
            g = self.potential.mass_enclosed(arr[:self.ndim], t=t*self.potential.units['time'])

    def test_circular_velocity(self):
        for arr,shp in zip(self.w0s, self._valu_return_shapes):
            g = self.potential.circular_velocity(arr[:self.ndim])
            assert g.shape == shp
            assert np.all(g > 0.)

            g = self.potential.circular_velocity(arr[:self.ndim], t=0.1)
            g = self.potential.circular_velocity(arr[:self.ndim], t=0.1*self.potential.units['time'])

            t = np.zeros(np.array(arr).shape[1:]) + 0.1
            g = self.potential.circular_velocity(arr[:self.ndim], t=t)
            g = self.potential.circular_velocity(arr[:self.ndim], t=t*self.potential.units['time'])

    def test_repr(self):
        pot_repr = repr(self.potential)
        if isinstance(self.potential.units, DimensionlessUnitSystem):
            assert "dimensionless" in pot_repr
        else:
            assert str(self.potential.units['length']) in pot_repr
            assert str(self.potential.units['time']) in pot_repr
            assert str(self.potential.units['mass']) in pot_repr

        for k in self.potential.parameters.keys():
            assert "{}=".format(k) in pot_repr

    def test_compare(self):
        # skip if composite potentials
        if len(self.potential.parameters) == 0:
            return

        other = self.potential.__class__(units=self.potential.units, **self.potential.parameters)
        assert other == self.potential

        pars = self.potential.parameters.copy()
        for k in pars.keys():
            if k != 0:
                pars[k] = 1.1*pars[k]
        other = self.potential.__class__(units=self.potential.units, **pars)
        assert other != self.potential

    def test_plot(self):
        p = self.potential

        if self.show_plots:
            f = p.plot_contours(grid=(np.linspace(-10., 10., 100), 0., 0.),
                                labels=["X"])
            # f.suptitle("slice off from 0., won't have cusp")
            # f.savefig(os.path.join(plot_path, "contour_x.png"))

            f = p.plot_contours(grid=(np.linspace(-10., 10., 100),
                                      np.linspace(-10., 10., 100),
                                      0.),
                                cmap='Blues')
            # f.savefig(os.path.join(plot_path, "contour_xy.png"))

            f = p.plot_contours(grid=(np.linspace(-10., 10., 100),
                                      1.,
                                      np.linspace(-10., 10., 100)),
                                cmap='Blues', labels=["X", "Z"])
            # f.savefig(os.path.join(plot_path, "contour_xz.png"))

            plt.show()
            plt.close('all')

    def test_save_load(self, tmpdir):
        """
        Test writing to a YAML file, and reading back in
        """
        fn = str(tmpdir.join("{}.yml".format(self.name)))
        self.potential.save(fn)
        p = load(fn)
        p.energy(self.w0[:self.w0.size//2])
        p.gradient(self.w0[:self.w0.size//2])

    def test_numerical_gradient_vs_gradient(self):
        """
        Check that the value of the implemented gradient function is close to a
        numerically estimated value. This is to check the coded-up version.
        """

        dx = 1E-3 * np.sqrt(np.sum(self.w0[:self.w0.size//2]**2))
        max_x = np.sqrt(np.sum([x**2 for x in self.w0[:self.w0.size//2]]))

        grid = np.linspace(-max_x,max_x,8)
        grid = grid[grid != 0.]
        grids = [grid for i in range(self.w0.size//2)]
        xyz = np.ascontiguousarray(np.vstack(map(np.ravel, np.meshgrid(*grids))).T)

        def energy_wrap(xyz):
            xyz = np.ascontiguousarray(xyz[None])
            return self.potential._energy(xyz, t=np.array([0.]))[0]

        num_grad = np.zeros_like(xyz)
        for i in range(xyz.shape[0]):
            num_grad[i] = np.squeeze([partial_derivative(energy_wrap, xyz[i], dim_ix=dim_ix, n=1, dx=dx, order=5)
                                      for dim_ix in range(self.w0.size//2)])
        grad = self.potential._gradient(xyz, t=np.array([0.]))

        assert np.allclose(num_grad, grad, rtol=self.tol)

    def test_orbit_integration(self):
        """
        Make we can integrate an orbit in this potential
        """
        w0 = self.w0
        w0 = np.vstack((w0,w0,w0)).T

        t1 = time.time()
        orbit = self.potential.integrate_orbit(w0, dt=1., n_steps=10000,
                                               Integrator=LeapfrogIntegrator)
        print("Integration time (10000 steps): {}".format(time.time() - t1))

        if self.show_plots:
            f = orbit.plot()
            f.suptitle("Vector w0")
            plt.show()
            plt.close(f)

        us = self.potential.units
        w0 = PhaseSpacePosition(pos=w0[:self.ndim]*us['length'],
                                vel=w0[self.ndim:]*us['length']/us['time'])
        orbit = self.potential.integrate_orbit(w0, dt=1., n_steps=10000,
                                               Integrator=LeapfrogIntegrator)

        if self.show_plots:
            f = orbit.plot()
            f.suptitle("Object w0")
            plt.show()
            plt.close(f)

    def test_pickle(self, tmpdir):
        fn = str(tmpdir.join("{}.pickle".format(self.name)))
        with open(fn, "wb") as f:
            pickle.dump(self.potential, f)

        with open(fn, "rb") as f:
            p = pickle.load(f)

        p.energy(self.w0[:self.w0.size//2])

class CompositePotentialTestBase(PotentialTestBase):
    @pytest.mark.skip(reason="Skip composite potential repr test")
    def test_repr(self):
        pass

    @pytest.mark.skip(reason="Skip composite potential compare test")
    def test_compare(self):
        pass
