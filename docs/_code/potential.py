import astropy.units as u
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

# example

import astropy.units as u
import numpy as np
import streamteam.potential as sp

usys = (u.kpc, u.Msun, u.Myr)
potential = sp.NFWPotential(v_h=(150*u.km/u.s).decompose(usys).value,
                            r_h=3., q1=1., q2=1., q3=1., usys=usys)

import streamteam.integrate as si
acc = lambda t,x: potential.acceleration(x)
integrator = si.LeapfrogIntegrator(acc)

x0 = np.array([[11.,6.,19.],[31.,0.,-4.]])
v0 = ([[50.,0.,0.],[120.,-120.,375.]]*u.km/u.s).decompose(usys).value
w0 = np.hstack((x0,v0))
t,ws = integrator.run(w0, dt=1., nsteps=10000)

import matplotlib.pyplot as plt
from matplotlib import cm
x = np.linspace(-50,50,200)
z = np.linspace(-50,50,200)
fig,ax = potential.plot_contours(grid=(x,0.,z), cmap=cm.gray_r)
ax.plot(ws[:,0,0], ws[:,0,2], marker=None, lw=2., alpha=0.6)
ax.plot(ws[:,1,0], ws[:,1,2], marker=None, lw=2., alpha=0.6)
fig.set_size_inches(8,8)
fig.savefig("../_static/potential/nfw.png")