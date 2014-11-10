import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import streamteam.potential as sp
import streamteam.integrate as si
from streamteam.units import galactic


# integrate & potential example
potential = sp.SphericalNFWPotential(v_h=(500*u.km/u.s).decompose(galactic).value,
                                     r_h=3., units=galactic)

x0 = np.array([[11.,6.,19.],[31.,0.,-4.]])
v0 = ([[50.,0.,0.],[70.,-70.,155.]]*u.km/u.s).decompose(galactic).value
w0 = np.hstack((x0,v0))
t,ws = potential.integrate_orbit(w0.copy(), dt=1., nsteps=10000)

x = np.linspace(-50,50,200)
z = np.linspace(-50,50,200)
fig,ax = potential.plot_contours(grid=(x,0.,z), cmap=cm.Blues)
ax.plot(ws[:,0,0], ws[:,0,2], marker=None, lw=1., alpha=0.75)
ax.plot(ws[:,1,0], ws[:,1,2], marker=None, lw=1., alpha=0.75, color='r')
fig.set_size_inches(6,6)
fig.savefig("../_static/examples/nfw.png")
