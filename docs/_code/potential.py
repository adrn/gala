import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import gary.potential as sp
from gary.units import galactic

# ----------------------------------------------------------------------------

p = sp.MiyamotoNagaiPotential(1E11, 6.5, 0.27, units=(u.kpc, u.Msun, u.Myr))

fig,axes = p.plot_contours(grid=(np.linspace(-15,15,100), 0., 1.), marker=None)
fig.set_size_inches(8,6)
fig.savefig("../_static/potential/miyamoto-nagai-1d.png")

xgrid = np.linspace(-15,15,100)
zgrid = np.linspace(-5,5,100)
fig,axes = p.plot_contours(grid=(xgrid, 1., zgrid))
fig.set_size_inches(8,6)
fig.savefig("../_static/potential/miyamoto-nagai-2d.png")


# ----------------------------------------------------------------------------
r_h = 20.
p = sp.SphericalNFWPotential(v_h=0.5, r_h=r_h, units=(u.kpc, u.Msun, u.Myr))
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

disk = sp.MiyamotoNagaiPotential(m=1E11, a=6.5, b=0.27, units=galactic)
bulge = sp.HernquistPotential(m=3E10, c=0.7, units=galactic)
pot = sp.CompositePotential(disk=disk, bulge=bulge)

x = z = np.linspace(-3.,3.,100)
fig,ax = pot.plot_contours(grid=(x,0,z), ax=ax)

fig.savefig("../_static/potential/composite.png")
