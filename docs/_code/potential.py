import matplotlib.pyplot as plt
import numpy as np
import streamteam.potential as sp

p = sp.MiyamotoNagaiPotential(1E11, 6.5, 0.27, usys=(u.kpc, u.Msun, u.Myr))

fig,axes = p.plot_contours(grid=(np.linspace(-15,15,100), 0., 1.), marker=None)
fig.set_size_inches(8,6)
fig.savefig("../_static/potential/miyamoto-nagai-1d.png")

xgrid = np.linspace(-15,15,100)
zgrid = np.linspace(-5,5,100)
fig,axes = p.plot_contours(grid=(xgrid, 1., zgrid))
fig.set_size_inches(8,6)
fig.savefig("../_static/potential/miyamoto-nagai-2d.png")