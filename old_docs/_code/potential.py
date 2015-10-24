import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import gary.potential as gp
from gary.units import galactic

# ----------------------------------------------------------------------------

p = gp.MiyamotoNagaiPotential(1E11, 6.5, 0.27, units=(u.kpc, u.Msun, u.Myr))

fig = p.plot_contours(grid=(np.linspace(-15,15,100), 0., 1.), marker=None)
fig.set_size_inches(8,6)
fig.savefig("../_static/potential/miyamoto-nagai-1d.png")

xgrid = np.linspace(-15,15,100)
zgrid = np.linspace(-5,5,100)
fig = p.plot_contours(grid=(xgrid, 1., zgrid))
fig.set_size_inches(8,6)
fig.savefig("../_static/potential/miyamoto-nagai-2d.png")


# ----------------------------------------------------------------------------
r_h = 20.
p = gp.SphericalNFWPotential(v_c=0.5*np.sqrt(np.log(2)-0.5), r_s=r_h, units=(u.kpc, u.Msun, u.Myr))
fig,ax = plt.subplots(1,1,figsize=(8,6))

r = np.zeros((100,3))
r[:,0] = np.logspace(np.log10(r_h/100.), np.log10(100*r_h), len(r))
menc = p.mass_enclosed(r)
ax.loglog(r/r_h, menc, marker=None)
ax.set_xlabel(r"$\log (r/r_s)$")
ax.set_ylabel(r"$M(<r)\,[{\rm M}_\odot]$")
fig.tight_layout()
fig.savefig("../_static/potential/mass-profile.png")

# ----------------------------------------------------------------------------

fig,ax = plt.subplots(1,1,figsize=(6,6))

disk = gp.MiyamotoNagaiPotential(m=1E11, a=6.5, b=0.27, units=galactic)
bulge = gp.HernquistPotential(m=3E10, c=0.7, units=galactic)
pot = gp.CompositePotential(disk=disk, bulge=bulge)

x = z = np.linspace(-3.,3.,100)
fig = pot.plot_contours(grid=(x,0,z), ax=ax)

fig.savefig("../_static/potential/composite.png")

# ----------------------------------------------------------------------------

def henon_heiles_funcs(units):
    def value(r, L):
        x,y = r.T
        return 0.5*(x**2 + y**2) + L*(x**2*y - y**3/3)

    def gradient(r, L):
        x,y = r.T
        grad = np.zeros_like(r)
        grad[...,0] = x + 2*L*x*y
        grad[...,1] = y + L*(x**2 - y**2)
        return grad

    def hessian(r, L):
        raise NotImplementedError()

    return value, gradient, hessian

class HenonHeilesPotential(gp.CartesianPotential):
    r"""
    The Henon-Heiles potential originally used to describe the non-linear
    motion of stars near the Galactic center.

    .. math::

        \Phi = \frac{1}{2}(x^2 + y^2) + \lambda(x^2 y - \frac{y^3}{3})

    Parameters
    ----------
    L : numeric
        Lambda parameter.
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.
    """

    def __init__(self, L, units=None):
        parameters = dict(L=L)
        func,gradient,hessian = henon_heiles_funcs(units)
        super(HenonHeilesPotential, self).__init__(func=func, gradient=gradient,
                                                   hessian=hessian,
                                                   parameters=parameters, units=units)

potential = HenonHeilesPotential(0.5)
t,w = potential.integrate_orbit([0.,0.,0.5,0.5], dt=0.03, nsteps=50000)

grid = np.linspace(-2,2,100)
fig = potential.plot_contours(grid=(grid,grid), levels=[0, 0.05,0.1,1/6.,0.5,1.,2,3,5],
                              cmap='Blues_r', subplots_kw=dict(figsize=(6,6)),
                              labels=['$x$','$y$'])
fig.axes[0].plot(w[:,0,0], w[:,0,1], marker='.',
                 linestyle='none', color='#fec44f', alpha=0.1)
fig.savefig("../_static/potential/henon-heiles.png")
