import copy
import pickle
import time

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.constants import G
from findiff import Diff

from gala._optional_deps import HAS_SYMPY
from gala.dynamics import PhaseSpacePosition
from gala.potential import Hamiltonian, StaticFrame
from gala.potential.potential.io import load
from gala.units import DimensionlessUnitSystem, UnitSystem


class PotentialTestBase:
    name = None
    potential = None  # MUST SET THIS
    frame = None
    tol = 1e-5
    show_plots = False

    sympy_hessian = True
    sympy_density = True
    check_finite_at_origin = True
    check_zero_at_infinity = True
    rotation = False

    skip_hessian = False
    skip_density = False

    @pytest.fixture(scope="class")
    def rng(self):
        return np.random.default_rng(42)

    def setup_method(self):
        # set up hamiltonian
        if self.frame is None:
            self.frame = StaticFrame(units=self.potential.units)
        self.H = Hamiltonian(self.potential, self.frame)
        self.rnd = np.random.default_rng(seed=42)

        cls = self.__class__
        if cls.name is None:
            cls.name = cls.__name__[4:]  # removes "Test"
        print(f"Testing potential: {cls.name}")
        self.w0 = np.array(self.w0)
        self.ndim = self.w0.size // 2

        # TODO: need to test also quantity objects and phasespacepositions!

        # these are arrays we will test the methods on:
        w0_2d = np.repeat(self.w0[:, None], axis=1, repeats=16)
        w0_3d = np.repeat(w0_2d[..., None], axis=2, repeats=8)
        w0_list = list(self.w0)
        w0_slice = w0_2d[:, :4]
        self.w0s = [self.w0, w0_2d, w0_3d, w0_list, w0_slice]
        self._grad_return_shapes = [
            (*self.w0[: self.ndim].shape, 1),
            w0_2d[: self.ndim].shape,
            w0_3d[: self.ndim].shape,
            (*self.w0[: self.ndim].shape, 1),
            w0_slice[: self.ndim].shape,
        ]
        self._hess_return_shapes = [
            (self.ndim, *self.w0[: self.ndim].shape, 1),
            (self.ndim, *w0_2d[: self.ndim].shape),
            (self.ndim, *w0_3d[: self.ndim].shape),
            (self.ndim, *self.w0[: self.ndim].shape, 1),
            (self.ndim, *w0_slice[: self.ndim].shape),
        ]
        self._valu_return_shapes = [x[1:] for x in self._grad_return_shapes]

    def test_unitsystem(self):
        assert isinstance(self.potential.units, UnitSystem)

        if isinstance(self.potential.units, DimensionlessUnitSystem):
            # Don't do a replace_units test for dimensionless potentials
            return

        # check that we can replace the units as expected
        usys = UnitSystem([u.pc, u.Gyr, u.radian, u.Msun])
        pot = copy.deepcopy(self.potential)

        pot2 = pot.replace_units(usys)
        assert pot2.units == usys
        assert pot.units == self.potential.units

    def test_energy(self):
        assert self.ndim == self.potential.ndim

        for arr, shp in zip(self.w0s, self._valu_return_shapes):
            v = self.potential.energy(arr[: self.ndim])
            assert v.shape == shp

            self.potential.energy(arr[: self.ndim], t=0.1)
            self.potential.energy(
                arr[: self.ndim], t=0.1 * self.potential.units["time"]
            )

            t = np.zeros(np.array(arr).shape[1:]) + 0.1
            self.potential.energy(arr[: self.ndim], t=t)
            self.potential.energy(arr[: self.ndim], t=t * self.potential.units["time"])

        if self.check_finite_at_origin:
            val = self.potential.energy([0.0, 0, 0])
            assert np.isfinite(val)

        if self.check_zero_at_infinity:
            val = self.potential.energy([1e12, 1e12, 1e12])
            np.testing.assert_allclose(val, 0.0, rtol=1e-05, atol=1e-6)

    def test_gradient(self):
        for arr, shp in zip(self.w0s, self._grad_return_shapes):
            g = self.potential.gradient(arr[: self.ndim])
            assert g.shape == shp

            g = self.potential.gradient(arr[: self.ndim], t=0.1)
            g = self.potential.gradient(
                arr[: self.ndim], t=0.1 * self.potential.units["time"]
            )

            t = np.zeros(np.array(arr).shape[1:]) + 0.1
            g = self.potential.gradient(arr[: self.ndim], t=t)
            g = self.potential.gradient(
                arr[: self.ndim], t=t * self.potential.units["time"]
            )

    def test_hessian(self):
        if self.skip_hessian:
            pytest.skip("Hessian not implemented for this potential")

        for arr, shp in zip(self.w0s, self._hess_return_shapes):
            g = self.potential.hessian(arr[: self.ndim])
            assert g.shape == shp

            g = self.potential.hessian(arr[: self.ndim], t=0.1)
            g = self.potential.hessian(
                arr[: self.ndim], t=0.1 * self.potential.units["time"]
            )

            t = np.zeros(np.array(arr).shape[1:]) + 0.1
            g = self.potential.hessian(arr[: self.ndim], t=t)
            g = self.potential.hessian(
                arr[: self.ndim], t=t * self.potential.units["time"]
            )

    def test_mass_enclosed(self):
        for arr, shp in zip(self.w0s, self._valu_return_shapes):
            g = self.potential.mass_enclosed(arr[: self.ndim])
            assert g.shape == shp
            assert np.all(g > 0.0)

            g = self.potential.mass_enclosed(arr[: self.ndim], t=0.1)
            g = self.potential.mass_enclosed(
                arr[: self.ndim], t=0.1 * self.potential.units["time"]
            )

            t = np.zeros(np.array(arr).shape[1:]) + 0.1
            g = self.potential.mass_enclosed(arr[: self.ndim], t=t)
            g = self.potential.mass_enclosed(
                arr[: self.ndim], t=t * self.potential.units["time"]
            )

    def test_circular_velocity(self):
        for arr, shp in zip(self.w0s, self._valu_return_shapes):
            g = self.potential.circular_velocity(arr[: self.ndim])
            assert g.shape == shp
            np.testing.assert_array_less(0.0, g)

            g = self.potential.circular_velocity(arr[: self.ndim], t=0.1)
            g = self.potential.circular_velocity(
                arr[: self.ndim], t=0.1 * self.potential.units["time"]
            )

            t = np.zeros(np.array(arr).shape[1:]) + 0.1
            g = self.potential.circular_velocity(arr[: self.ndim], t=t)
            g = self.potential.circular_velocity(
                arr[: self.ndim], t=t * self.potential.units["time"]
            )

    def test_repr(self):
        pot_repr = repr(self.potential)
        if isinstance(self.potential.units, DimensionlessUnitSystem):
            assert "dimensionless" in pot_repr
        else:
            assert str(self.potential.units["length"]) in pot_repr
            assert str(self.potential.units["time"]) in pot_repr
            assert str(self.potential.units["mass"]) in pot_repr

        for k in self.potential.parameters:
            assert f"{k}=" in pot_repr

    def test_compare(self):
        # skip if composite potentials
        if len(self.potential.parameters) == 0:
            return

        other = self.potential.__class__(
            units=self.potential.units, **self.potential.parameters
        )
        assert other == self.potential

        pars = self.potential.parameters.copy()
        for k in pars:
            if isinstance(pars[k], str | int):
                continue

            pars[k] = pars[k] * 1.1  # fmt: skip, ruff: noqa
        other = self.potential.__class__(units=self.potential.units, **pars)
        assert other != self.potential

        # check that comparing to non-potentials works
        assert self.potential != "sup"
        assert self.potential is not None

    def test_plot(self):
        p = self.potential

        p.plot_contours(grid=(np.linspace(-10.0, 10.0, 100), 0.0, 0.0), labels=["X"])

        p.plot_contours(
            grid=(
                np.linspace(-10.0, 10.0, 100),
                np.linspace(-10.0, 10.0, 100),
                0.0,
            ),
            cmap="Blues",
        )

        p.plot_contours(
            grid=(
                np.linspace(-10.0, 10.0, 100),
                1.0,
                np.linspace(-10.0, 10.0, 100),
            ),
            cmap="Blues",
            labels=["X", "Z"],
        )

        _f, _a = p.plot_rotation_curve(R_grid=np.linspace(0.1, 10.0, 100))

        plt.close("all")

        if self.show_plots:
            plt.show()

    def test_save_load(self, tmpdir):
        """
        Test writing to a YAML file, and reading back in
        """
        fn = str(tmpdir.join(f"{self.name}.yml"))
        self.potential.save(fn)
        p = load(fn)
        p.energy(self.w0[: self.w0.size // 2])
        p.gradient(self.w0[: self.w0.size // 2])

    def test_numerical_gradient_vs_gradient(self, rng):
        """
        Check that the value of the implemented gradient function is close to a
        numerically estimated value. This is to check the coded-up version.
        """

        # NOTE: 1e-3 and 2 are magic numbers and should maybe be configurable
        w0_r = np.linalg.norm(self.w0[: self.potential.ndim])
        dx = 1e-3 * w0_r
        max_x = 2 * w0_r

        # Pick random points in 3-space, build a finite-difference grid around each
        # point to compute numerical gradient
        N_points = 16
        pt_xyz = rng.uniform(-max_x, max_x, size=(self.potential.ndim, N_points))

        grid = np.arange(-4, 4 + 1, 1) * dx
        grids = np.meshgrid(*[grid for _ in range(self.potential.ndim)], indexing="ij")
        grid_xyz = np.stack(grids, axis=0)

        d_dxs = [Diff(i, dx) for i in range(self.potential.ndim)]
        for n in range(N_points):
            pt_grid = (
                np.expand_dims(
                    pt_xyz[:, n], axis=tuple(np.arange(1, 1 + self.potential.ndim))
                )
                + grid_xyz
            )

            energy = self.potential.energy(pt_grid).value

            dPhi_dx = [d_dx(energy) for d_dx in d_dxs]
            num_grad = np.stack(dPhi_dx, axis=0)
            grad = self.potential.gradient(pt_grid).value

            assert np.allclose(grad, num_grad, rtol=self.tol)

    def test_numerical_density_vs_density(self, rng):
        """
        Compare a numerically estimated Laplacian (trace of Hessian) times 4*pi*G
        to the implemented density function via Poisson's equation
        """
        if self.skip_density:
            pytest.skip("density not implemented for this potential")

        # TODO: duplicate code here to test_numerical_gradient_vs_gradient; refactor

        _G = G if not isinstance(self.potential.units, DimensionlessUnitSystem) else 1.0

        # NOTE: 1e-3 and 2 are magic numbers and should maybe be configurable
        w0_r = np.linalg.norm(self.w0[: self.potential.ndim])
        dx = 1e-3 * w0_r
        max_x = 2 * w0_r

        # Pick random points in 3-space, build a finite-difference grid around each
        # point to compute numerical gradient
        N_points = 16
        pt_xyz = rng.uniform(-max_x, max_x, size=(self.potential.ndim, N_points))

        grid = np.arange(-4, 4 + 1, 1) * dx
        grids = np.meshgrid(*[grid for _ in range(self.potential.ndim)], indexing="ij")
        grid_xyz = np.stack(grids, axis=0)

        d2_dx2s = [Diff(i, dx) ** 2 for i in range(self.potential.ndim)]
        for n in range(N_points):
            pt_grid = (
                np.expand_dims(
                    pt_xyz[:, n], axis=tuple(np.arange(1, 1 + self.potential.ndim))
                )
                + grid_xyz
            )

            energy = self.potential.energy(pt_grid).value

            d2Phi_dx2 = [d2_dx2(energy) for d2_dx2 in d2_dx2s]
            num_Lap = np.sum(d2Phi_dx2, axis=0)
            dens_Lap = (
                (self.potential.density(pt_grid) * 4 * np.pi * _G)
                .decompose(self.potential.units)
                .value
            )

            print("YO", np.abs((num_Lap - dens_Lap) / dens_Lap).max())

            # NOTE: 1e2 factor here is also a magic number
            assert np.allclose(dens_Lap, num_Lap, rtol=1e2 * self.tol)

    def test_hessian_density_consistency(self):
        """
        Check that the trace of the Hessian matches the density via Poisson's equation
        """
        if self.skip_hessian or self.skip_density:
            pytest.skip("Hessian not implemented for this potential")

        _G = G if not isinstance(self.potential.units, DimensionlessUnitSystem) else 1.0

        for arr in self.w0s:
            hess = self.potential.hessian(arr[: self.ndim])
            lap = np.sum(np.diagonal(hess, axis1=0, axis2=1), axis=-1)

            dens = self.potential.density(arr[: self.ndim])
            rho_from_hess = lap / (4.0 * np.pi * _G)
            print("YO", dens, rho_from_hess, np.diagonal(hess, axis1=0, axis2=1))
            assert u.allclose(dens, rho_from_hess, rtol=self.tol)

    def test_orbit_integration(self, t1=0.0, t2=1000.0, nsteps=10000):
        """
        Make sure we can integrate an orbit in this potential
        """
        w0 = self.w0
        w0 = np.vstack((w0, w0, w0)).T

        dt = (t2 - t1) / nsteps

        twall = time.time()
        orbit = self.H.integrate_orbit(w0, t1=t1, dt=dt, n_steps=nsteps)
        print(f"Integration time ({nsteps} steps): {time.time() - twall}")

        if self.show_plots:
            f = orbit.plot()
            f.suptitle("Vector w0")
            plt.show()
            plt.close(f)

        us = self.potential.units
        w0 = PhaseSpacePosition(
            pos=w0[: self.ndim] * us["length"],
            vel=w0[self.ndim :] * us["length"] / us["time"],
        )
        orbit = self.H.integrate_orbit(w0, t1=t1, dt=dt, n_steps=nsteps)

        if self.show_plots:
            f = orbit.plot()
            f.suptitle("Object w0")
            plt.show()
            plt.close(f)

    def test_pickle(self, tmpdir):
        fn = str(tmpdir.join(f"{self.name}.pickle"))
        with open(fn, "wb") as f:
            pickle.dump(self.potential, f)

        with open(fn, "rb") as f:
            p = pickle.load(f)

        p.energy(self.w0[: self.w0.size // 2])

    @pytest.mark.skipif(not HAS_SYMPY, reason="requires sympy to run this test")
    def test_against_sympy(self):
        # TODO: should really split this into separate tests for each check...

        import sympy as sy
        from sympy import Q

        # compare Gala gradient, hessian, and density to sympy values

        pot = self.potential
        Phi, v, p = pot.to_sympy()

        # Derive sympy gradient and hessian functions to evaluate:
        from scipy.special import gamma, gammainc

        def lowergamma(a, x):
            # Differences between scipy and sympy lower gamma
            return gammainc(a, x) * gamma(a)

        modules = [
            {
                "atan": np.arctan,
                # "lowergamma": lowergamma,
                "gamma": gamma,
                "re": np.real,
                "im": np.imag,
            },
            "numpy",
            "scipy",
            "sympy",
        ]

        vars_ = list(p.values()) + list(v.values())
        np.bitwise_and.reduce([Q.real(x) for x in vars_])
        # Phi = sy.refine(Phi, assums)
        e_func = sy.lambdify(vars_, Phi, modules=modules)

        if self.sympy_density:
            dens_tmp = sum(sy.diff(Phi, var, 2) for var in v.values()) / (
                4 * sy.pi * p["G"]
            )
            # dens_tmp = sy.refine(dens_tmp, assums)
            dens_func = sy.lambdify(vars_, dens_tmp, modules=modules)

        grad = sy.derive_by_array(Phi, list(v.values()))
        # grad = sy.refine(grad, assums)
        grad_func = sy.lambdify(vars_, grad, modules=modules)

        if self.sympy_hessian:
            Hess = sy.hessian(Phi, list(v.values()))
            # Hess = sy.refine(Hess, assums)
            Hess_func = sy.lambdify(vars_, Hess, modules=modules)

        # Make a dict of potential parameter values without units:
        par_vals = {}
        for k, v in pot.parameters.items():
            par_vals[k] = v.value

        N = 64  # MAGIC NUMBER:
        trial_x = self.rnd.uniform(-10.0, 10.0, size=(pot.ndim, N))
        x_dict = dict(zip(["x", "y", "z"], trial_x))

        f_gala = pot.energy(trial_x).value
        f_sympy = e_func(G=pot.G, **par_vals, **x_dict)
        e_close = np.allclose(f_gala, f_sympy)
        test_cases = [e_close]
        vals = [(f_gala, f_sympy)]

        if self.sympy_density:
            d_gala = pot.density(trial_x).value
            d_sympy = dens_func(G=pot.G, **par_vals, **x_dict)
            d_close = np.allclose(d_gala, d_sympy)
            test_cases.append(d_close)
            vals.append((d_gala, d_sympy))

        G_gala = pot.gradient(trial_x).value
        G_sympy = grad_func(G=pot.G, **par_vals, **x_dict)
        g_close = np.allclose(G_gala, G_sympy)
        test_cases.append(g_close)
        vals.append((G_gala, G_sympy))

        if self.sympy_hessian:
            H_gala = pot.hessian(trial_x).value
            H_sympy = Hess_func(G=pot.G, **par_vals, **x_dict)
            h_close = np.allclose(H_gala, H_sympy)
            test_cases.append(h_close)
            vals.append((H_gala, H_sympy))

        if not all(test_cases):
            names = ["energy", "density", "gradient", "hessian"]
            for name, (val1, val2), test in zip(names, vals, test_cases):
                if not test:
                    print(trial_x)
                    print(f"{pot}: {name}\nGala:{val1}\nSympy:{val2}")

        assert all(test_cases)

    def test_regression_165(self):
        if self.potential.ndim == 1:
            pytest.skip("ndim = 1")

        with pytest.raises(ValueError):
            self.potential.energy(8.0)

        with pytest.raises(ValueError):
            self.potential.gradient(8.0)

        with pytest.raises(ValueError):
            self.potential.circular_velocity(8.0)

    @pytest.mark.parametrize("meth", ["energy", "gradient", "density"])
    def test_rotation_shift(self, meth):
        if not self.rotation:
            pytest.skip("Rotation has no impact for this potential")
        if meth == "density" and not self.sympy_density:
            pytest.skip("No analytic density")

        x = np.array([10.0, 5.0, 3.0])
        x0 = np.array([1.0, 1.0, 3.0])
        R = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        R_pot = self.potential.replicate(R=R)
        origin_pot = self.potential.replicate(origin=x0)
        R_origin_pot = self.potential.replicate(R=R, origin=x0)

        x_R = getattr(R_pot, meth)(x)
        test_val = getattr(self.potential, meth)(R @ x)
        if test_val.size > 1:
            test_val = R.T @ test_val
        assert u.allclose(x_R, test_val)

        x_origin = getattr(origin_pot, meth)(x)
        assert u.allclose(x_origin, getattr(self.potential, meth)(x - x0))

        x_R_origin = getattr(R_origin_pot, meth)(x)
        test_val = getattr(self.potential, meth)(R @ (x - x0))
        if test_val.size > 1:
            test_val = R.T @ test_val
        assert u.allclose(x_R_origin, test_val)


class CompositePotentialTestBase(PotentialTestBase):
    @pytest.mark.skip(reason="Skip composite potential repr test")
    def test_repr(self):
        pass

    @pytest.mark.skip(reason="Skip composite potential compare test")
    def test_compare(self):
        pass

    @pytest.mark.skip(reason="to_sympy() not implemented yet")
    def test_against_sympy(self):
        pass

    @pytest.mark.parametrize("meth", ["energy", "gradient", "density"])
    def test_rotation_shift(self, meth):
        if not self.rotation:
            pytest.skip("Rotation has no impact for this potential")

        x = np.array([10.0, 5.0, 3.0])
        x0 = np.array([1.0, 1.0, 3.0])
        R = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        R_pot = self.potential.__class__()
        origin_pot = self.potential.__class__()
        R_origin_pot = self.potential.__class__()
        for k, p in self.potential.items():
            R_pot[k] = p.replicate(R=R)
            origin_pot[k] = p.replicate(origin=x0)
            R_origin_pot[k] = p.replicate(R=R, origin=x0)

        x_R = getattr(R_pot, meth)(x)
        test_val = getattr(self.potential, meth)(R @ x)
        if test_val.size > 1:
            test_val = R.T @ test_val
        assert u.allclose(x_R, test_val)

        x_origin = getattr(origin_pot, meth)(x)
        assert u.allclose(x_origin, getattr(self.potential, meth)(x - x0))

        x_R_origin = getattr(R_origin_pot, meth)(x)
        test_val = getattr(self.potential, meth)(R @ (x - x0))
        if test_val.size > 1:
            test_val = R.T @ test_val
        assert u.allclose(x_R_origin, test_val)
