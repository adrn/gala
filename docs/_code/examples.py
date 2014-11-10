import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import streamteam.potential as sp
import streamteam.integrate as si
from streamteam.units import galactic

# integrate & potential example
v_c = (200*u.km/u.s).decompose(galactic).value
potential = sp.SphericalNFWPotential(v_c=v_c, r_s=10., units=galactic)

# single orbit
initial_conditions = np.array([10., 0, 0, 0, v_c, 0])
t,orbit = potential.integrate_orbit(initial_conditions, dt=0.5, nsteps=10000)

# multiple orbits
norbits = 1000
stddev = [0.1,0.1,0.1,0.01,0.01,0.01]  # 100 pc spatial scale, ~10 km/s velocity scale
initial_conditions = np.random.normal(initial_conditions, stddev, size=(norbits,6))
t,orbits = potential.integrate_orbit(initial_conditions, dt=0.5, nsteps=10000)

fig,ax = plt.subplots(1,1,figsize=(6,6))
ax.plot(orbits[-1,:,0], orbits[-1,:,1], marker='.', linestyle='none',
        alpha=0.75, color='#cc0000')

x = y = np.linspace(-15,15,100)
potential.plot_contours(grid=(x,y,0), ax=ax, cmap=cm.Greys)
fig.set_size_inches(6,6)
fig.savefig("../_static/examples/nfw.png")
