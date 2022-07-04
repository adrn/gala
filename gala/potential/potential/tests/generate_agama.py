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
    agama_pot = agama.Potential(file=str(this_path / 'pot_disk_506151.pot'))

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
    tbl['xyz'] = test_xyz.T * u.kpc
    tbl['pot'] = pot * (u.km/u.s)**2
    tbl['acc'] = acc * (u.km/u.s)**2 / u.kpc

    tbl.write(this_path / 'agama_cylspline_test.fits')


if __name__ == '__main__':
    main()
