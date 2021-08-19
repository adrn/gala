
def make(static_path):

    # orbits-in-derail.rst
    import astropy.units as u
    import gala.dynamics as gd
    import gala.potential as gp
    from gala.units import galactic

    pot = gp.PlummerPotential(m=1E10 * u.Msun, b=1. * u.kpc, units=galactic)
    w0 = gd.PhaseSpacePosition(pos=[2., 0, 0] * u.kpc,
                               vel=[0., 75, 15] * u.km/u.s)
    orbit = gp.Hamiltonian(pot).integrate_orbit(w0, dt=1., n_steps=5000)

    # animation 1:
    fig, anim = orbit[:1000].animate(stride=10)
    anim.save(static_path / 'orbit-anim1.mp4')

    # animation 2:
    fig, anim = orbit[:1000].cylindrical.animate(components=['rho', 'z'],
                                                 stride=10)
    anim.save(static_path / 'orbit-anim2.mp4')
