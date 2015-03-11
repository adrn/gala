# coding: utf-8

""" Test action-angle stuff """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import OrderedDict
import logging
import os
import re

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
                                 "../../../test-data/"))

logger.setLevel(logging.DEBUG)


def test_simple_f():
    true_ws = 2*np.pi*np.array([0.581, 0.73])
    true_as = np.array([5*(np.cos(np.radians(15.)) + 1j*np.sin(np.radians(15.))),
                        1.8*(np.cos(np.radians(85.)) + 1j*np.sin(np.radians(85.)))])
    true_A = np.sqrt(true_as.imag**2 + true_as.real**2)
    true_phi = np.arctan2(true_as.imag, true_as.real)

    logger.info("")
    logger.info("True ω = {}".format(true_ws.tolist()))
    logger.info("True a = {}".format(true_as.tolist()))
    logger.info("True A = {}".format(true_A.tolist()))

    ts = [np.linspace(0., 300., 12000),
          np.linspace(0., 300., 24414),
          np.linspace(150., 450., 12000),
          np.linspace(150., 450., 24414),
          np.linspace(0., 300., 12000) + 50*(2*np.pi/true_ws[0])]

    for i,t in enumerate(ts):
        print(i)
        f = np.sum(true_as[None] * np.exp(1j * true_ws[None] * t[:,None]), axis=1)

        naff = gd.NAFF(t, debug=True, debug_path=os.path.join(plot_path,'naff-debug'))

        # try recovering the strongest frequency
        w = naff.frequency(f[:naff.n])
        np.testing.assert_allclose(true_ws[0], w, atol=1E-6)

        # try recovering all frequencies
        output = naff.frecoder(f[:naff.n], break_condition=1E-5)
        nu,A,phi = output
        np.testing.assert_allclose(true_ws, nu[:2], atol=1E-7)
        np.testing.assert_allclose(true_A, A[:2], atol=1E-4)
        np.testing.assert_allclose(true_phi, phi[:2], atol=1E-4)

def test_simple_f2():
    true_ws = 2*np.pi*np.array([0.581, -0.73, 0.91])
    true_as = np.array([5*(np.cos(np.radians(35.)) + 1j*np.sin(np.radians(35.))),
                        1.8*(np.cos(np.radians(75.)) + 1j*np.sin(np.radians(75.))),
                        0.7*(np.cos(np.radians(45.)) + 1j*np.sin(np.radians(45.)))])
    true_A = np.sqrt(true_as.imag**2 + true_as.real**2)
    true_phi = np.arctan2(true_as.imag, true_as.real)

    logger.info("")
    logger.info("True ω = {}".format(true_ws.tolist()))
    logger.info("True a = {}".format(true_as.tolist()))
    logger.info("True A = {}".format(true_A.tolist()))

    ts = [np.linspace(0., 300., 12000),
          np.linspace(0., 300., 24414),
          np.linspace(150., 450., 12000),
          np.linspace(150., 450., 24414),
          np.linspace(0., 300., 12000) + 50*(2*np.pi/true_ws[0])]

    for i,t in enumerate(ts):
        print(i)
        print("{} periods".format(t.max() / (2*np.pi/true_ws)))
        f = np.sum(true_as[None] * np.exp(1j * true_ws[None] * t[:,None]), axis=1)

        naff = gd.NAFF(t, debug=True, debug_path=os.path.join(plot_path,'naff-debug'))

        # try recovering the strongest frequency
        w = naff.frequency(f[:naff.n])
        np.testing.assert_allclose(true_ws[0], w, atol=1E-6)

        # try recovering all frequencies
        output = naff.frecoder(f[:naff.n], break_condition=1E-4)
        nu,A,phi = output
        np.testing.assert_allclose(true_ws, nu[:3], atol=1E-7)
        np.testing.assert_allclose(true_A, A[:3], atol=1E-4)
        np.testing.assert_allclose(true_phi, phi[:3], atol=1E-4)

# ----------------------------------------------------------------------------

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

def test_error_estimate():
    potential = gp.LogarithmicPotential(v_c=np.sqrt(2.), r_h=0.1,
                                        q1=1., q2=1., q3=0.9,
                                        units=galactic)
    w0 = [1., 0.1, 0.2, 0., 1., 0.]
    t,w = potential.integrate_orbit(w0, dt=0.005, nsteps=32768,
                                    Integrator=DOPRI853Integrator)
    logger.info("Done integrating orbit")
    fig = gd.plot_orbits(w, linestyle='none', alpha=0.1)
    fig.savefig(os.path.join(plot_path,"orbit_error_estimate.png"))

    fs = poincare_polar(w[:,0])

    naff = NAFF(t)
    f,d,ixes = naff.find_fundamental_frequencies(fs, nintvec=30, break_condition=None)
    logger.info("Solved for fundamental freqs")

    fprimes = []
    for i in range(3):
        _d = d[d['n'] == i]
        fp = _d['A'][:,np.newaxis] * np.exp(1j * _d['freq'][:,np.newaxis] * t[np.newaxis])
        fp = np.sum(fp, axis=0)
        fprimes.append(fp)

        # plotting...
        fig,axes = plt.subplots(2,2,figsize=(14,14),sharex='col',sharey='col')

        alpha = 0.1
        axes[0,0].plot(fs[i].real, fs[i].imag, linestyle='none', alpha=alpha)
        axes[1,0].plot(fp.real, fp.imag, linestyle='none', alpha=alpha)

        axes[0,1].plot(fs[i].real)
        axes[1,1].plot(fp.real)
        axes[1,1].set_xlim(0,1000)
        fig.savefig(os.path.join(plot_path,"error_estimate_fprime_{}.png".format(i)))

    # dp = double prime
    fdp,ddp,ixesdp = naff.find_fundamental_frequencies(fprimes, nintvec=30, break_condition=None)
    logger.info("Solved for fundamental freqs of f'(t)")

    df = f - fdp
    logger.info("δω = {}".format(df))
    assert np.abs(df).max() < 1E-6

class LaskarBase(object):
    potential = None

    def read_files(self):
        this_path = os.path.join(test_data_path, 'papapilippou-orbits', self.name)

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

        f,d,ixes = naff.find_fundamental_frequencies(fs[:2], nintvec=15, break_condition=None)
        nvecs = naff.find_integer_vectors(f, d)

        # TEST
        this_d = d[d['n'] == 0]
        print(this_d['|A|'])
        print(self.true_tables[0]['A'])

        phi = coord.Angle(this_d['phi']*u.radian).wrap_at(360*u.deg).to(u.degree).value
        print(phi)
        print(self.true_tables[0]['phi'])

        return

        fprime = this_d['A'][:,np.newaxis] * np.exp(1j * this_d['freq'][:,np.newaxis] * self.t[np.newaxis])
        fprime = np.sum(fprime, axis=0)

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

class TestLaskarBox1(LaskarBase):

    def setup_class(self):
        self.name = 'box-orbit-1'
        self.dt = 0.004
        self.nsteps = 2**16

class TestLaskarLoop1xy(LaskarBase):

    def setup_class(self):
        self.name = 'loop-orbit-1-xy'
        self.dt = 0.01
        self.nsteps = 2**15

class TestLaskarLoop2xy(LaskarBase):

    def setup_class(self):
        self.name = 'loop-orbit-2-xy'
        self.dt = 0.01
        self.nsteps = 2**15

class TestLaskarLoop2rphi(LaskarBase):

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

        L = np.cross(self.w[:,0,:3], self.w[:,0,3:])
        Lz = L[:,2].mean()
        print("Lz", Lz)

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
        self.nsteps = 20000

class TestLogarithmic3D(NAFFVsSandersBase):

    def setup_class(self):
        self.name = 'logarithmic-3d'
        self.potential = gp.LogarithmicPotential(v_c=0.24, r_h=10.,
                                                 q1=1., q2=0.9, q3=0.7, units=galactic)
        self.w0 = np.array([10.,0.,0.,0.02,0.25,-0.02])
        self.dt = 2.5
        self.nsteps = 40000

# -------------------------------------------------------------------------------------
class MonicaBase(object):

    def setup(self):
        self.path = os.path.join(test_data_path, 'monica-naff')

        with open(os.path.join(self.path, 'filenames')) as f:
            self.filenames = [l.strip() for l in f.readlines()]

        self.index = self.filenames.index(self.filename)

        split_indices = []
        with open(os.path.join(self.path, 'adrian1.int')) as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                if re.search('[\*]+', line) is not None:
                    split_indices.append(i)

        if self.index == 0:
            these_lines = lines[:split_indices[0]]
        else:
            these_lines = lines[split_indices[self.index-1]+2:split_indices[self.index]]

        freqs = OrderedDict()
        freqs['freq'] = []
        freqs['l'] = []
        freqs['m'] = []
        freqs['n'] = []
        freqs['nq'] = []
        nq = None
        for line in these_lines:
            s = re.search('\s*[-]+\s+([XYZ])', line)
            if s is not None:
                nq = 'xyz'.index(s.groups()[0].lower())
                continue

            if nq is None:
                continue

            freqs['freq'].append(float(line.split()[1]))
            freqs['l'].append(int(line.split()[2]))
            freqs['m'].append(int(line.split()[3]))
            freqs['n'].append(int(line.split()[4]))
            freqs['nq'].append(nq)

        names,formats = [],[]
        for k,v in freqs.items():
            names.append(k)
            formats.append(type(v[0]))
        self.monica_freqs = np.array(zip(*freqs.values()), dtype=dict(names=names,formats=formats))

    def test_naff(self):
        # read in the orbit and compute the frequencies with my implementation
        t = np.loadtxt(os.path.join(self.path, self.filename), usecols=[0])
        w = np.loadtxt(os.path.join(self.path, self.filename), usecols=range(1,7))

        naff = NAFF(t)

        # compute complex time series from orbit
        fs = [(w[:,i] + 1j*w[:,i+3]) for i in range(3)]
        f,d,ixes = naff.find_fundamental_frequencies(fs, nintvec=15)
        nvecs = naff.find_integer_vectors(f, d)

        for nq in range(3):
            this_d = d[d['n'] == nq]
            nvec = nvecs[d['n'] == nq]
            monica = self.monica_freqs[self.monica_freqs['nq'] == nq]
            # print(np.vstack((freq,nvec.T)).T)

            for j in range(5):
                row = monica[j]
                print(row.dtype.names)
                print("APW: {}, {}, {}, {}".format(this_d[j]['freq'], this_d[j]['|A|'], this_d[j]['phi'], nvec[j]))
                print("Monica: {}".format(row['freq']))
                print()

            return

            break

        # print(f)
        # print(self.monica_freqs)

class TestInterAxisTube(MonicaBase):
    def setup_class(self):
        self.filename = 'inter-axis-tube.txt'

class TestShortAxisTube(MonicaBase):
    def setup_class(self):
        self.filename = 'short-axis-tube.txt'

# ------------------------------------------------------------------------------

def test_weird_bump():
    t = np.load(os.path.join(test_data_path, "naff/t.npy"))
    w = np.load(os.path.join(test_data_path, "naff/w.npy"))

    every = 2

    # i1,i2 = (0,200001)
    # naff = NAFF(t[i1:i2:every], debug=True, debug_path=os.path.join(test_data_path, "naff"))
    # fs = poincare_polar(w[i1:i2:every])
    # f,d,ixes = naff.find_fundamental_frequencies(fs, nintvec=5)
    # print(f)

    # freqs,d,ixes,is_tube = gd.naff.orbit_to_freqs(t[i1:i2:every], w[i1:i2:every],
    #                                               silently_fail=False, nintvec=5)
    # print(freqs)
    # print()

    # ----

    i1,i2 = (350000,550001)
    naff = NAFF(t[i1:i2:every], debug=True, debug_path=os.path.join(test_data_path, "naff"))
    fs = poincare_polar(w[i1:i2:every])
    f,d,ixes = naff.find_fundamental_frequencies(fs[2:3], nintvec=5)
    print(f)

    freqs,d,ixes,is_tube = gd.naff.orbit_to_freqs(t[i1:i2:every], w[i1:i2:every],
                                                  silently_fail=False, nintvec=5)
    print(freqs)
