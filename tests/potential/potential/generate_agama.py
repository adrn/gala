import pathlib

import astropy.table as at
import astropy.units as u
import numpy as np

this_path = pathlib.Path(__file__).absolute().parent


def main():
    # For pytest:
    import agama

    agama.setUnits(mass=1, length=1, time=1)

    # Shared by Ana Bonaca
    agama_pot = agama.Potential(file=str(this_path / "pot_disk_506151.pot"))

    # Generate a grid of points to evaluate at:
    test_R = np.linspace(0, 150, 128)
    test_z = np.linspace(-100, 100, 128)
    test_Rz = np.stack(list(map(np.ravel, np.meshgrid(test_R, test_z))))
    test_xyz = np.zeros((3, test_Rz.shape[1]))
    test_xyz[0] = test_Rz[0]
    test_xyz[2] = test_Rz[1]

    pot = agama_pot.potential(test_xyz.T)[:, None]
    acc = agama_pot.force(test_xyz.T)

    tbl = at.QTable()
    tbl["xyz"] = test_xyz.T * u.kpc
    tbl["pot"] = pot * (u.km / u.s) ** 2
    tbl["acc"] = acc * (u.km / u.s) ** 2 / u.kpc

    tbl.write(this_path / "agama_cylspline_test.fits")


# Modern Agama CylSpline exports (with derivative blocks and, for mmax > 0, one block
# per azimuthal harmonic). Each is saved as a .pot file plus a table of reference
# values at random positions inside and outside of the grid.
CYLSPLINE_FIXTURES = {
    # axisymmetric, z-symmetric
    "mn": (
        {"type": "MiyamotoNagai", "mass": 1e10, "scaleRadius": 3.0, "scaleHeight": 0.3},
        {"mmax": 0},
    ),
    # triaxial: even m only, z-symmetric (z grid starts at 0)
    "triaxial": (
        {
            "type": "Dehnen",
            "mass": 1e11,
            "scaleRadius": 2.0,
            "gamma": 1.0,
            "axisRatioY": 0.6,
            "axisRatioZ": 0.4,
        },
        {"mmax": 4},
    ),
    # tilted triaxial: no symmetry, all m, full z grid
    "tilted": (
        {
            "type": "Dehnen",
            "mass": 1e11,
            "scaleRadius": 2.0,
            "gamma": 1.0,
            "axisRatioY": 0.6,
            "axisRatioZ": 0.4,
            "orientation": [0.3, 0.4, 0.2],
        },
        {"mmax": 3},
    ),
}


def main_cylspline_fixtures():
    import agama

    agama.setUnits(mass=1, length=1, velocity=1)  # Msun, kpc, km/s

    rng = np.random.default_rng(8675309)
    for name, (pot_kw, cyl_kw) in CYLSPLINE_FIXTURES.items():
        cyl = agama.Potential(
            type="CylSpline",
            potential=agama.Potential(**pot_kw),
            gridSizeR=24,
            gridSizez=24,
            rmin=0.1,
            rmax=100.0,
            zmin=0.05,
            zmax=100.0,
            **cyl_kw,
        )
        cyl.export(str(this_path / f"agama_cylspline_{name}.pot"))

        # mostly inside the grid, plus some points outside of it
        xyz = np.concatenate(
            (
                rng.uniform(-40, 40, size=(256, 3)) * [1, 1, 0.25],
                rng.uniform(-300, 300, size=(32, 3)),
            )
        )
        pot, force, deriv = cyl.eval(xyz, pot=True, acc=True, der=True)

        tbl = at.QTable()
        tbl["xyz"] = xyz * u.kpc
        tbl["pot"] = pot * (u.km / u.s) ** 2
        tbl["acc"] = force * (u.km / u.s) ** 2 / u.kpc
        # dF/dx, dF/dy, dF/dz components in Agama order: xx, yy, zz, xy, yz, zx
        tbl["force_deriv"] = deriv * (u.km / u.s) ** 2 / u.kpc**2
        tbl["dens"] = cyl.density(xyz) * u.Msun / u.kpc**3
        tbl.write(this_path / f"agama_cylspline_{name}.fits", overwrite=True)


if __name__ == "__main__":
    import sys

    if "cylspline" in sys.argv[1:]:
        main_cylspline_fixtures()
    else:
        main()
