"""
Generate test data for EXP interface tests
"""

__all__ = ["EXPTestDataGenerator"]

import os
import pathlib

import astropy.table as at
import astropy.units as u
import numpy as np
import pyEXP

import gala.potential as gp
from gala.units import SimulationUnitSystem, galactic

this_path = pathlib.Path(__file__).parent


class EXPTestDataGenerator:
    def __init__(
        self,
        potential: gp.PotentialBase,
        name: str | None = None,
        lmax: int = 4,
        nmax: int = 10,
        overwrite: bool = False,
    ):
        if name is None:
            name = potential.__class__.__name__[:-9]
        self.name = str(name)
        self.gala_pot = potential

        length_unit = mass_unit = None
        for name, val in self.gala_pot.parameters.items():
            if val.unit.is_equivalent(u.kpc):
                length_unit = val
            elif val.unit.is_equivalent(u.Msun):
                mass_unit = val

        if length_unit is None or mass_unit is None:
            msg = "Potential must have length and mass unit parameters."
            raise ValueError(msg)
        self.usys = SimulationUnitSystem(mass=mass_unit, length=length_unit, G=1)
        self.overwrite = overwrite

        self.lmax = lmax
        self.nmax = nmax
        l_size = (self.lmax + 1) * (self.lmax + 2) // 2
        self.coef_shape = (l_size, self.nmax)

    def make_empirical_basis(self, r_grid: u.Quantity | None = None):
        if r_grid is None:
            r_grid = np.geomspace(1e-3, 2e2, 1024) * u.kpc

        xyz_grid = np.zeros((3, r_grid.size)) * u.kpc
        xyz_grid[0] = r_grid

        self._basis_table_file = this_path / f"{self.name}.model"

        if self._basis_table_file.exists() and not self.overwrite:
            print(
                f"File {self._basis_table_file} already exists. Use overwrite=True to "
                "regenerate."
            )
            return None

        tbl = at.Table()
        tbl["r"] = r_grid.decompose(self.usys).value
        tbl["density"] = self.gala_pot.density(xyz_grid).decompose(self.usys).value
        tbl["mass"] = self.gala_pot.mass_enclosed(xyz_grid).decompose(self.usys).value
        tbl["energy"] = self.gala_pot.energy(xyz_grid).decompose(self.usys).value
        tbl.meta["comments"] = ["! r density mass energy", f"{len(tbl)}"]
        tbl.write(
            self._basis_table_file,
            format="ascii.no_header",
            delimiter=" ",
            overwrite=True,
            comment="",
        )
        return tbl

    def make_config(self, basis_tbl: at.Table):
        self._cache_file = this_path / f"{self.name}.cache"
        if self._cache_file.exists() and self.overwrite:
            self._cache_file.unlink()

        bconfig = f"""
---
id: sphereSL
parameters :
  numr: {len(basis_tbl)}
  rmin: {basis_tbl["r"].min():.4f}
  rmax: {basis_tbl["r"].max():.1f}
  Lmax: {self.lmax}
  nmax: {self.nmax}
  modelname: {self.name}.model
  cachename: {self.name}.cache
...
        """
        print(bconfig)
        self._basis_file = this_path / f"{self.name}-basis.yml"
        with open(self._basis_file, "w", encoding="utf-8") as f:
            f.write(bconfig)

        cwd = os.getcwd()
        os.chdir(this_path)
        basis = pyEXP.basis.Basis.factory(bconfig)
        os.chdir(cwd)
        return basis

    def make_coef(
        self,
        basis: pyEXP.basis.Basis,
        coef_arr: np.ndarray,
        time: u.Quantity | float,
        coefs: pyEXP.coefs.Coefs | None = None,
    ):
        if hasattr(time, "unit"):
            time = time.to_value(self.usys["time"])

        # Create coefficients with a dummy particle at time=0
        coef = basis.createFromArray([1.0], [[1.0], [1.0], [1.0]], time=time)

        # Set values for the coefficients based on input array
        coef.assign(coef_arr, self.lmax, self.nmax)

        if coefs is None:
            coefs = pyEXP.coefs.Coefs.makecoefs(coef, self.name)
        coefs.add(coef)

        return coefs

    def save_coefs(
        self,
        coefs: pyEXP.coefs.Coefs,
        filename: str | pathlib.Path | None = None,
    ):
        if filename is None:
            filename = this_path / f"{self.name}-coefs.h5"
        filename = pathlib.Path(filename)

        if filename.exists() and not self.overwrite:
            print(f"File {filename} already exists. Use overwrite=True to regenerate.")
            return str(filename)
        if filename.exists():
            filename.unlink()

        coefs.WriteH5Coefs(str(filename))
        return filename


def main():
    """
    This generates an empirical basis from a spherical Hernquist potential, and makes
    two coefficient files: one with a single snapshot at time=0 and one with multiple
    snapshots at different times.
    """
    # Random parameter values:
    pot = gp.HernquistPotential(m=1.25234e11, c=3.845, units=galactic)

    gen = EXPTestDataGenerator(pot, overwrite=True, name="EXP-Hernquist")

    # make the empirical basis:
    r_grid = np.geomspace(1e-3, 2e2, 1024) * u.kpc
    tbl = gen.make_empirical_basis(r_grid)
    basis = gen.make_config(tbl)

    coef_arr = np.zeros(gen.coef_shape, dtype=np.complex128)
    coef_arr[0, 0] = 2.05  # close to matching the input Hernquist potential

    coefs = gen.make_coef(basis, coef_arr, time=0.0 * u.Myr)
    gen.save_coefs(coefs, this_path / f"{gen.name}-single-coefs.hdf5")

    # A few snapshots with slightly different coefficients:
    coefs = gen.make_coef(basis, coef_arr, time=0.0 * u.Myr)

    coef_arr = np.zeros(gen.coef_shape, dtype=np.complex128)
    coef_arr[0, 0] = 2.1
    coefs = gen.make_coef(basis, coef_arr, time=500.0 * u.Myr, coefs=coefs)

    coef_arr = np.zeros(gen.coef_shape, dtype=np.complex128)
    coef_arr[0, 0] = 2.15
    coefs = gen.make_coef(basis, coef_arr, time=1000.0 * u.Myr, coefs=coefs)

    coef_arr = np.zeros(gen.coef_shape, dtype=np.complex128)
    coef_arr[0, 0] = 2.2
    coefs = gen.make_coef(basis, coef_arr, time=1500.0 * u.Myr, coefs=coefs)

    coef_arr = np.zeros(gen.coef_shape, dtype=np.complex128)
    coef_arr[0, 0] = 2.25
    coefs = gen.make_coef(basis, coef_arr, time=2000.0 * u.Myr, coefs=coefs)
    gen.save_coefs(coefs, this_path / f"{gen.name}-multi-coefs.hdf5")


if __name__ == "__main__":
    main()
