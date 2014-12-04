# coding: utf-8

""" Test action-angle stuff """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
from astropy import log as logger
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelmax

# Project
from ..naff import NAFF, poincare_polar
from ...integrate import DOPRI853Integrator
from ... import potential as gp
from ... import dynamics as gd
from ...units import galactic

plot_path = "plots/tests/dynamics/naff"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

logger.important("Plots located at: {}".format(plot_path))

# TODO: config item to specify path to test data?
test_data_path = os.path.abspath(os.path.join(os.path.split(__file__)[0],
                                 "../../../test-data/papapilippou-orbits"))

def estimate_axisym_freqs(t, w):
    R = np.sqrt(w[:,0,0]**2 + w[:,0,1]**2)
    phi = np.arctan2(w[:,0,1], w[:,0,0])
    z = w[:,0,2]

    ix = argrelmax(R)[0]
    fR = np.mean(2*np.pi / (t[ix[1:]] - t[ix[:-1]]))
    ix = argrelmax(phi)[0]
    fphi = np.mean(2*np.pi / (t[ix[1:]] - t[ix[:-1]]))
    ix = argrelmax(z)[0]
    fz = np.mean(2*np.pi / (t[ix[1:]] - t[ix[:-1]]))

    return fR, fphi, fz

class LaskarBase(object):

    def read_files(self):
        this_path = os.path.join(test_data_path, self.name)

        # read initial conditions
        d = np.loadtxt(os.path.join(this_path, "ics.txt"))
        w0 = d[:6]
        poincare = bool(d[6])

        # read true tables
        tables = []
        for i in range(3):
            fn = os.path.join(this_path, "{}.txt".format(i))
            try:
                tbl = np.genfromtxt(fn, names=True)
            except:
                continue

            tables.append(tbl)

        self.w0 = w0
        self.poincare = poincare
        self.true_tables = tables

    def setup(self):
        from pytest import config

        logger.debug("-"*50)
        logger.debug(self.name)

        # read tables of true values and initial conditions
        self.read_files()

        # potential is the same for all PL96 tables
        self.potential = gp.LogarithmicPotential(v_c=np.sqrt(2.), r_h=0.1,
                                                 q1=1., q2=0.9, q3=1., units=galactic)

        orbit_filename = os.path.join(plot_path,"orbit_{}.npy".format(self.name))
        if os.path.exists(orbit_filename) and config.option.overwrite:
            os.remove(orbit_filename)

        if not os.path.exists(orbit_filename):
            logger.info("Integrating...")
            logger.debug("dt : {}".format(self.dt))
            logger.debug("Nsteps : {}".format(self.nsteps))
            t,w = self.potential.integrate_orbit(self.w0, dt=self.dt,
                                                 nsteps=self.nsteps,
                                                 Integrator=DOPRI853Integrator)
            np.save(orbit_filename, np.vstack((w.T,t[np.newaxis,np.newaxis])))

        logger.info("Orbit loaded")
        wt = np.load(orbit_filename)
        self.t = wt[-1,0].copy()
        self.w = wt[:-1].T.copy()

        # plot and save orbit
        fig = gd.plot_orbits(self.w, marker='.', linestyle='none', alpha=0.2)
        fig.savefig(os.path.join(plot_path,"orbit_{}.png".format(self.name)))

        if self.w.shape[-1] == 6:
            # plot energy conservation
            E = self.potential.total_energy(self.w[:,0,:3],self.w[:,0,3:])
            plt.clf()
            plt.semilogy(self.t[1:], np.abs(E[1:]-E[0]), marker=None)
            plt.savefig(os.path.join(plot_path,"energy_{}.png".format(self.name)))

    def test_naff(self):
        naff = NAFF(self.t)

        # complex time series
        if self.poincare:
            logger.info("Using Poincaré polar coordinates")
            fs = poincare_polar(self.w[:,0])
        else:
            logger.info("Using Cartesian coordinates")
            fs = [(self.w[:,0,i] + 1j*self.w[:,0,i+3]) for i in range(3)]

        f,d,ixes = naff.find_fundamental_frequencies(fs, nintvec=15)

        # TODO: compare with true tables
        for i in range(3):
            if sum(d['n'] == i) == 0 or i == len(self.true_tables):
                break

            this_d = d[d['n'] == i]
            this_tbl = self.true_tables[i]

            dfreq = np.abs((this_d['freq'] - this_tbl['freq'])/this_tbl['freq'])
            dA = np.abs((this_d['|A|'] - this_tbl['A'])/this_tbl['A'])

            import astropy.coordinates as coord
            import astropy.units as u
            phi = coord.Angle(this_d['phi']*u.deg).wrap_at(360*u.deg).value
            dphi = (phi - this_tbl['phi'])

            print(dfreq)
            print(dA)
            print(dphi)

        return

        logger.important("True freqs: {}".format(self.true_freqs))
        logger.important("Find freqs: {}".format(f))

        actions,angles,freqs = gd.find_actions(self.t, self.w[:,0], N_max=6, units=galactic)
        logger.important("Sanders freqs: {}".format(freqs))

        if np.all(np.isfinite(self.true_freqs)):
            done = []
            for freq in f:
                for i,tfreq in enumerate(self.true_freqs):
                    if i not in done:
                        if abs(abs(tfreq) - abs(freq)) < 1E-3:
                            done.append(i)
            assert len(done) == len(self.true_freqs)

        if len(f) < 2:
            return

        nvecs = naff.find_integer_vectors(f, d)

        return

        Js = np.zeros(3)
        for row,nvec in zip(d,nvecs):
            a = row['|A|']*np.exp(1j*row['phi'])
            print(a.real, a.imag)
            Js[0] += nvec[0] * nvec.dot(f) * row['|A|']**2
            Js[1] += nvec[1] * nvec.dot(f) * row['|A|']**2
            Js[2] += nvec[2] * nvec.dot(f) * row['|A|']**2
            # Js[0] += nvec[0] * nvec.dot(f) * a.real**2
            # Js[1] += nvec[1] * nvec.dot(f) * a.real**2
            # Js[2] += nvec[2] * nvec.dot(f) * a.real**2

        if hasattr(self.potential, 'action_angle'):
            true_Js,angles = self.potential.action_angle(self.w[...,:3], self.w[...,3:])
            true_Js = np.mean(true_Js[:,0], axis=0)

        print(Js)
        print(true_Js)

        # a = d['|A|']*np.exp(1j*d['phi'])
        # a.real, a.imag

class TestTable1(LaskarBase):

    def setup_class(self):
        self.name = 'box-orbit-1'
        self.dt = 0.005
        self.nsteps = 2**15



# # -------------------------------------------------------------------------------------
# # Hand-constructed time-series
# #
# class TestHandConstructed(NAFFBase):
#     def setup(self):
#         Ts = np.array([1., 1.2, 1.105])
#         As = (1., 0.5, 0.2)
#         self.t = np.linspace(0,100,50000)
#         self.true_freqs = (2*np.pi) / Ts[0:1]

#         f = np.sum([A*(np.cos(2*np.pi*self.t/T) + 1j*np.sin(2*np.pi*self.t/T)) for T,A in zip(Ts,As)], axis=0)
#         z = np.zeros_like(f.real)
#         self.w = np.vstack((f.real,z,z,f.imag,z,z)).T[:,np.newaxis]

#         self.dt = 0.1
#         self.nsteps = 20000

# # -------------------------------------------------------------------------------------
# # Harmonic Oscillator
# #
# class TestHarmonicOscillator(NAFFBase):
#     def setup_class(self):
#         Ts = np.array([1., 1.2, 1.105])
#         self.true_freqs = 2*np.pi/Ts
#         self.potential = gp.HarmonicOscillatorPotential(self.true_freqs)
#         self.w0 = np.array([1,0,0.2,0.,0.1,-0.8])

#         self.dt = 0.009
#         self.nsteps = 20000

# # -------------------------------------------------------------------------------------
# # Logarithmic potential, 1D orbit as in Papaphilippou & Laskar (1996), Table 2
# #
# class TestLogarithmic1D(NAFFBase):
#     def setup_class(self):
#         self.true_freqs = np.array([2.13905125])
#         self.potential = gp.LogarithmicPotential(v_c=np.sqrt(2.), r_h=0.1,
#                                                  q1=1., q2=0.9, q3=1., units=galactic)
#         self.w0 = np.array([0.49,0.,0.,1.4,0.,0.])

#         self.dt = 0.005
#         self.nsteps = 2**15

# # -------------------------------------------------------------------------------------
# # Logarithmic potential, 2D Box orbit as in Papaphilippou & Laskar (1996), Table 1
# #
# class TestLogarithmic2DBox(NAFFBase):
#     def setup_class(self):
#         self.true_freqs = np.array([2.16326132, 3.01405257])
#         self.potential = gp.LogarithmicPotential(v_c=np.sqrt(2.), r_h=0.1,
#                                                  q1=1., q2=0.9, q3=1., units=galactic)
#         self.w0 = np.array([0.49,0.,0.,1.3156,0.4788,0.])

#         self.dt = 0.005
#         self.nsteps = 2**15

# # -------------------------------------------------------------------------------------
# # Logarithmic potential, 2D Loop orbit as in Papaphilippou & Laskar (1996), Table 3
# #
# class TestLogarithmic2DLoop(NAFFBase):
#     def setup_class(self):
#         self.true_freqs = np.array([2.94864765, 1.35752682])
#         self.potential = gp.LogarithmicPotential(v_c=np.sqrt(2.), r_h=0.1,
#                                                  q1=1., q2=0.9, q3=1., units=galactic)
#         self.w0 = np.array([0.49,0.,0.,0.4788,1.3156,0.])

#         self.dt = 0.005
#         self.nsteps = 2**15

# # -------------------------------------------------------------------------------------
# # Logarithmic potential, 3D Loop orbit
# #
# class TestLogarithmic3DLoop(NAFFBase):
#     def setup_class(self):
#         self.true_freqs = np.array([np.nan, np.nan, np.nan])
#         self.potential = gp.LogarithmicPotential(v_c=0.24, r_h=10.,
#                                                  q1=1., q2=0.9, q3=0.7, units=galactic)
#         self.w0 = np.array([10.,0.,0.,0.07,0.22,-0.07])

#         self.dt = 2.5
#         self.nsteps = 2**13

# # -------------------------------------------------------------------------------------
# # Triaxial NFW potential, 3D Loop orbit
# #
# class TestFlattenedNFWLoop(NAFFBase):
#     def setup_class(self):
#         self.true_freqs = np.array([np.nan, np.nan, np.nan])
#         self.potential = gp.LeeSutoTriaxialNFWPotential(v_c=0.22, r_s=30.,
#                                                         a=1., b=1., c=0.8, units=galactic)
#         self.w0 = np.array([1.,20,0,0.05,0,0.185])

#         self.dt = 2.
#         self.nsteps = 40000

# # -------------------------------------------------------------------------------------
# # Triaxial NFW potential, 3D Loop orbit
# #
# class TestTriaxialNFWLoop(NAFFBase):
#     def setup_class(self):
#         self.true_freqs = np.array([np.nan, np.nan, np.nan])
#         self.potential = gp.LeeSutoTriaxialNFWPotential(v_c=0.22, r_s=30.,
#                                                         a=1., b=0.9, c=0.7, units=galactic)
#         self.w0 = np.array([0,30,0,0.025,0,0.2])

#         self.dt = 4.
#         self.nsteps = 20000


# # -------------------------------------------------------------------------------------
# # Logarithmic potential, 2D Loop orbit as in Papaphilippou & Laskar (1996), Table 3
# #
# class TestAPW(NAFFBase):
#     def setup_class(self):
#         # self.true_freqs = np.array([2.94864765, 1.35752682])
#         self.true_freqs = np.array([np.nan, np.nan, np.nan])
#         self.potential = gp.LogarithmicPotential(v_c=np.sqrt(2.), r_h=0.1,
#                                                  q1=1., q2=1., q3=0.9, units=galactic)
#         self.w0 = np.array([0.49,0.,0.,0.4788,1.3156,0.1])

#         self.dt = 0.002
#         self.nsteps = 2**16
