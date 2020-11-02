""" Test dynamics plotting functions """

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
        psps.append(PhaseSpacePosition(pos=np.random.random(size=(3, 16)),
                                       vel=np.random.random(size=(3, 16))))
        psps.append(PhaseSpacePosition(pos=np.random.random(size=(3, 16))*u.kpc,
                                       vel=np.random.random(size=(3, 16))*u.km/u.s))

        orbits = []
        orbits.append(Orbit(pos=np.random.random(size=(3, 16)),
                            vel=np.random.random(size=(3, 16)),
                            t=np.linspace(0, 1, 16)))
        orbits.append(Orbit(pos=np.random.random(size=(3, 16, 2)),
                            vel=np.random.random(size=(3, 16, 2)),
                            t=np.linspace(0, 1, 16)))
        orbits.append(Orbit(pos=np.random.random(size=(3, 16))*u.kpc,
                            vel=np.random.random(size=(3, 16))*u.km/u.s,
                            t=np.linspace(0, 1, 16)*u.Myr))

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
        fig, axes = plt.subplots(1, 2)
        fig = plot_projections(x[:2], autolim=True, axes=axes,  # noqa
                               subplots_kwargs=dict(sharex=True),
                               labels=['x', 'y'],
                               plot_function=plt.plot,
                               marker='o', linestyle='--', color='r')
