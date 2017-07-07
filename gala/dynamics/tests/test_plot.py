# coding: utf-8

""" Test dynamics plotting functions """

from __future__ import division, print_function

# Third-party
import numpy as np
import astropy.units as u

# Project
from ..core import PhaseSpacePosition
from ..orbit import Orbit
from ..plot import plot_projections

class TestPlotting(object):

    def setup(self):

        psps = []
        psps.append(PhaseSpacePosition(pos=np.random.random(size=3),
                                       vel=np.random.random(size=3)))
        psps.append(PhaseSpacePosition(pos=np.random.random(size=(3,16)),
                                       vel=np.random.random(size=(3,16))))
        psps.append(PhaseSpacePosition(pos=np.random.random(size=(3,16))*u.kpc,
                                       vel=np.random.random(size=(3,16))*u.km/u.s))

        orbits = []
        orbits.append(Orbit(pos=np.random.random(size=(3,16)),
                            vel=np.random.random(size=(3,16)),
                            t=np.linspace(0,1,16)))
        orbits.append(Orbit(pos=np.random.random(size=(3,16,2)),
                            vel=np.random.random(size=(3,16,2)),
                            t=np.linspace(0,1,16)))
        orbits.append(Orbit(pos=np.random.random(size=(3,16))*u.kpc,
                            vel=np.random.random(size=(3,16))*u.km/u.s,
                            t=np.linspace(0,1,16)*u.Myr))

        self.psps = psps
        self.orbits = orbits

    def test_plot_projections(self):
        import matplotlib.pyplot as plt

        # TODO: need major test improvements here
        # let the function create the figure
        for p in self.psps:
            _ = p.plot()

        for o in self.orbits:
            _ = o.plot()

        x = self.psps[0].xyz.value
        fig,axes = plt.subplots(1,2)
        fig = plot_projections(x[:2], autolim=True, axes=axes,
                               subplots_kwargs=dict(sharex=True),
                               labels=['x', 'y'],
                               plot_function=plt.plot,
                               marker='o', linestyle='--', color='r')


# def test_orbits(tmpdir):

#     # generate an "orbit"
#     n = 8
#     t = np.linspace(0, 100, 1000).reshape(1000,1)
#     x = np.cos(np.random.uniform(1.,8.,size=(1,n))*t).T[None]
#     y = np.sin(np.random.uniform(1.,8.,size=(1,n))*t).T[None]
#     z = np.cos(np.random.uniform(1.,8.,size=(1,n))*t).T[None]
#     vx = vy = vz = np.zeros_like(x)
#     w = np.rollaxis(np.vstack((x,y,z,vx,vy,vz)), -1, 1)

#     fig = plot_orbits(w, linestyle='none', marker='.', alpha=0.25)
#     fig.savefig(os.path.join(str(tmpdir), "all_orbits.png"))

#     fig = plot_orbits(w, ix=0, linestyle='none', marker='.', alpha=0.25)
#     fig.savefig(os.path.join(str(tmpdir), "one_orbit.png"))

#     fig = plot_orbits(w, ix=0, linestyle='none', marker='.', alpha=0.25,
#                       labels=("herp","derp","merp"))
#     fig.savefig(os.path.join(str(tmpdir), "one_orbit_labels.png"))

#     fig = plot_orbits(w, triangle=True, linestyle='-', marker=None)
#     fig.savefig(os.path.join(str(tmpdir), "all_orbits_triangle.png"))

#     fig = plot_orbits(w, ix=0, triangle=True, linestyle='-', marker=None)
#     fig.savefig(os.path.join(str(tmpdir), "one_orbit_triangle.png"))

#     fig = plot_orbits(w, ix=0, triangle=True, linestyle='-', marker=None,
#                       labels=("herp","derp","merp"))
#     fig.savefig(os.path.join(str(tmpdir), "one_orbit_triangle_labels.png"))

# def test_three_panel(tmpdir):

#     q = np.random.uniform(0.,10.,size=(1000,3))
#     q0 = np.array([5,5,5])

#     fig = three_panel(q)
#     fig.savefig(os.path.join(str(tmpdir), "three-panel-random.png"))

#     fig = three_panel(q, triangle=True)
#     fig.savefig(os.path.join(str(tmpdir), "three-panel-random_triangle.png"))

#     fig = three_panel(q, relative_to=q0, labels=[r'$\Omega_1$',r'$\Omega_2$',r'$\Omega_3$'])
#     fig.savefig(os.path.join(str(tmpdir), "three-panel-random-relative.png"))

#     fig = three_panel(q, relative_to=q0, triangle=True, labels=[r'$\Omega_1$',r'$\Omega_2$',r'$\Omega_3$'])
#     fig.savefig(os.path.join(str(tmpdir), "three-panel-random-relative_triangle.png"))

# def test_1d(tmpdir):

#     t = np.linspace(0,100.,1000)
#     q = np.cos(2*np.pi*t/10.)[None]

#     fig = plot_orbits(q, labels=(r"$\theta$",))
#     fig.savefig(os.path.join(str(tmpdir), "1d-orbit-labels.png"))

#     fig = plot_orbits(q, t=t, labels=(r"$\theta$",))
#     fig.savefig(os.path.join(str(tmpdir), "1d-orbit-labels-time.png"))

# def test_2d(tmpdir):

#     t = np.linspace(0,100.,1000)

#     q = np.zeros((2,len(t)))
#     q[0,:] = np.cos(2*np.pi*t/10.)
#     q[1,:] = np.sin(2*np.pi*t/5.5)

#     fig = plot_orbits(q, labels=(r"$\theta$",r"$\omega$"))
#     fig.savefig(os.path.join(str(tmpdir), "2d-orbit-labels.png"))
