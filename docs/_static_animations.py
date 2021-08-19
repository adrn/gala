def make_orbit_animations(static_path):

    # orbits-in-derail.rst
    import astropy.units as u
    import gala.dynamics as gd
    import gala.potential as gp
    from gala.units import galactic

    file1 = static_path / 'orbit-anim1.mp4'
    file2 = static_path / 'orbit-anim2.mp4'

    if file1.exists() and file2.exists():
        print("Orbit animations exist - skipping...")
        return

    pot = gp.PlummerPotential(m=1E10 * u.Msun, b=1. * u.kpc, units=galactic)
    w0 = gd.PhaseSpacePosition(pos=[2., 0, 0] * u.kpc,
                               vel=[0., 75, 15] * u.km/u.s)
    orbit = gp.Hamiltonian(pot).integrate_orbit(w0, dt=1., n_steps=5000)

    # animation 1:
    fig, anim = orbit[:1000].animate(stride=10)
    anim.save(file1)

    # animation 2:
    fig, anim = orbit[:1000].cylindrical.animate(components=['rho', 'z'],
                                                 stride=10)
    anim.save(file2)


if __name__ == "__main__":
    import pathlib
    make_orbit_animations(pathlib.Path('./_static').resolve().absolute())
