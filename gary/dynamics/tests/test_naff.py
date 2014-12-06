# coding: utf-8

""" Test action-angle stuff """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import astropy.coordinates as coord
import astropy.units as u
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
    potential = None

    def read_files(self):
        this_path = os.path.join(test_data_path, self.name)

        # read initial conditions
        d = np.loadtxt(os.path.join(this_path, "ics.txt"))
        w0 = d[:6]
        poincare = bool(d[6])
        self.q = d[7]
        self.true_freqs = d[8:]

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
        if self.potential is None:
            self.potential = gp.LogarithmicPotential(v_c=np.sqrt(2.), r_h=0.1,
                                                     q1=1., q2=self.q, q3=1.,
                                                     units=galactic)

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

        f,d,ixes = naff.find_fundamental_frequencies(fs[:2], nintvec=15)
        nvecs = naff.find_integer_vectors(f, d)

        # TODO: compare with true tables
        for i in range(2):
            if sum(d['n'] == i) == 0 or i == len(self.true_tables):
                break

            this_d = d[d['n'] == i]
            this_nvecs = nvecs[d['n'] == i]
            this_tbl = self.true_tables[i][:len(this_d)]

            dfreq = np.abs((this_d['freq'] - this_tbl['freq'])/this_tbl['freq'])
            dA = np.abs((this_d['|A|'] - this_tbl['A'])/this_tbl['A'])
            dn = this_nvecs[:,0] - this_tbl['n']
            dm = this_nvecs[:,1] - this_tbl['m']

            phi = coord.Angle(this_d['phi']*u.deg).wrap_at(360*u.deg).value
            dphi = (phi - this_tbl['phi'])

            print("∆Freq (percent):", 100*dfreq)
            print("∆A (percent):", 100*dA)
            print("∆φ (abs):", dphi)
            print("∆n:", dn)
            print("∆m:", dm)
            print()
            print("freq:", this_d['freq'])
            print("A:", this_d['|A|'])
            print("φ:", this_d['phi'])
            print("-"*79)

        # Compare the true frequencies
        done = np.zeros_like(self.true_freqs).astype(bool)
        for i,true_f in enumerate(self.true_freqs):
            for ff in f:
                if np.abs(np.abs(ff) - np.abs(true_f)) < 1E-4:
                    done[i] = True
                    break

        assert np.all(done)

class TestBox1(LaskarBase):

    def setup_class(self):
        self.name = 'box-orbit-1'
        self.dt = 0.005
        self.nsteps = 2**15

class TestLoop1xy(LaskarBase):

    def setup_class(self):
        self.name = 'loop-orbit-1-xy'
        self.dt = 0.01
        self.nsteps = 2**15

class TestLoop2xy(LaskarBase):

    def setup_class(self):
        self.name = 'loop-orbit-2-xy'
        self.dt = 0.01
        self.nsteps = 2**15

class TestLoop2rphi(LaskarBase):

    def setup_class(self):
        self.name = 'loop-orbit-2-rphi'
        self.dt = 0.01
        self.nsteps = 2**15

# -------------------------------------------------------------------------------------

class NAFFVsSandersBase(LaskarBase):

    def read_files(self):
        pass

    def test_naff(self):
        naff = NAFF(self.t)

        # complex time series
        logger.info("Using Poincaré polar coordinates")
        fs = poincare_polar(self.w[:,0])

        f,d,ixes = naff.find_fundamental_frequencies(fs, nintvec=15, break_condition=None)
        nvecs = naff.find_integer_vectors(f, d)

        max_T = np.max(np.abs(2*np.pi/f))
        int_time = self.dt*self.nsteps
        logger.important("Integrated for ~{} periods".format(int_time/max_T))

        # compure sanders frequencies
        s_actions,s_angles,s_freqs = gd.find_actions(self.t[::2], self.w[::2],
                                                     N_max=6, units=galactic)

        ptp_freqs = estimate_axisym_freqs(self.t, self.w)

        print("NAFF:", f)
        print("Sanders:", s_freqs)
        print("ptp:", ptp_freqs)

        # compute frequnecy diffusion with NAFF
        naff = NAFF(self.t[:self.nsteps//2+1])

        # first window
        fs = poincare_polar(self.w[:self.nsteps//2+1,0])
        f1,d1,ixes1 = naff.find_fundamental_frequencies(fs, nintvec=15, break_condition=None)
        fs = poincare_polar(self.w[self.nsteps//2:,0])
        f2,d2,ixes2 = naff.find_fundamental_frequencies(fs, nintvec=15, break_condition=None)

        dfreq = f2 - f1
        print("Window 1 freqs:", f1)
        print("Window 2 freqs:", f2)
        print("Abs. freq change:", dfreq)
        print("Percent freq change:", np.abs(dfreq/f1)*100)

        # compare NAFF with Sanders
        # done = np.zeros(3).astype(bool)
        # for i,true_f in enumerate(s_freqs):
        #     for ff in f:
        #         if np.abs(np.abs(ff) - np.abs(true_f)) < 1E-4:
        #             done[i] = True
        #             break
        # assert np.all(done)

        # try reconstructing actions with NAFF
        Js = np.zeros(3)
        for row,nvec in zip(d,nvecs):
            Js[0] += nvec[0] * nvec.dot(f) * row['|A|']**2
            Js[1] += nvec[1] * nvec.dot(f) * row['|A|']**2
            Js[2] += nvec[2] * nvec.dot(f) * row['|A|']**2

        print(Js)
        print(s_actions)

class TestFlattenedNFW(NAFFVsSandersBase):

    def setup_class(self):
        self.name = 'flattened-nfw'
        self.potential = gp.LeeSutoTriaxialNFWPotential(v_c=0.22, r_s=30.,
                                                        a=1., b=1., c=0.8, units=galactic)
        self.w0 = np.array([30.,2,5,0,0.16,0.05])
        self.dt = 2.
        self.nsteps = 50000

class TestLogarithmic3D(NAFFVsSandersBase):

    def setup_class(self):
        self.name = 'logarithmic-3d'
        self.potential = gp.LogarithmicPotential(v_c=0.24, r_h=10.,
                                                 q1=1., q2=0.9, q3=0.7, units=galactic)
        self.w0 = np.array([10.,0.,0.,0.02,0.25,-0.02])
        self.dt = 2.5
        self.nsteps = 40000
