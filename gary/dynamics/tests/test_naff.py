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
import pytest

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

def test_cy_naff():
    from .._naff import naff_frequency

    t = np.linspace(0., 300., 12000)
    naff = gd.NAFF(t)

    true_ws = 2*np.pi*np.array([0.581, 0.73])
    true_as = np.array([5*(np.cos(np.radians(15.)) + 1j*np.sin(np.radians(15.))),
                        1.8*(np.cos(np.radians(85.)) + 1j*np.sin(np.radians(85.)))])
    f = np.sum(true_as[None] * np.exp(1j * true_ws[None] * t[:,None]), axis=1)

    ww = naff_frequency(true_ws[0], naff.tz, naff.chi,
                        np.ascontiguousarray(f.real),
                        np.ascontiguousarray(f.imag),
                        naff.T)
    np.testing.assert_allclose(ww, true_ws[0], atol=1E-8)

# TODO: make a class to do these tests for given arrays of freqs, complex amps.

class SimpleBase(object):

    """ Need to define:

            self.amp
            self.omega

        in subclass setup().
    """
    def setup(self):
        print("PAReNT SETUP")
        self.A = np.sqrt(self.amp.imag**2 + self.amp.real**2)
        self.phi = np.arctan2(self.amp.imag, self.amp.real)

    def make_f(self, t):
        a = self.amp
        w = self.omega
        return np.sum(a[None] * np.exp(1j * w[None] * t[:,None]), axis=1)

    def test_T_scaling(self):
        # time arrays
        # end_T = np.linspace(32, 512, 25)
        end_T = np.logspace(np.log10(32), np.log10(512), 251)
        ts = [np.arange(0., ee, 0.01) for ee in end_T]

        dws = []
        for i,t in enumerate(ts):
            f = self.make_f(t)

            naff = gd.NAFF(t, p=self.p)

            # try recovering the strongest frequency
            w = naff.frequency(f)
            dw = self.omega[0] - w
            dws.append(dw)

        plt.clf()
        plt.loglog(end_T, np.abs(dws))
        derp = np.linspace(end_T.min(), end_T.max(), 1000)
        plt.loglog(derp, derp**(-4.), marker=None)
        plt.xlabel("T")
        plt.ylabel(r"$\delta \omega$")
        plt.savefig(os.path.join(plot_path, "{0}_T_scaling.png".format(self.__class__.__name__)))

    def test_dt_scaling(self):
        # time arrays
        dt = 2**np.linspace(-4, 0, 25)
        ts = [np.arange(0., 150, ee) for ee in dt]

        dws = []
        for i,t in enumerate(ts):
            f = self.make_f(t)

            # create NAFF object for this time array
            naff = gd.NAFF(t, p=self.p)

            # try recovering the strongest frequency
            w = naff.frequency(f)
            dw = self.omega[0] - w
            dws.append(dw)

        plt.clf()
        plt.loglog(dt, np.abs(dws))
        # derp = np.linspace(dt.min(), dt.max(), 1000)
        plt.xlabel("dt")
        plt.ylabel(r"$\delta \omega$")
        plt.xlim(dt.min(), dt.max())
        plt.ylim(1E-7,1E-5)
        plt.savefig(os.path.join(plot_path, "{0}_dt_scaling.png".format(self.__class__.__name__)))

    def test_rolling_window(self):
        # ts = [np.linspace(0.+dd, 150.+dd, 42104) for dd in np.linspace(0,150,250)]
        ts = [np.linspace(0.+dd, 150.+dd, 42104) for dd in np.linspace(0,10,150)]
        dws = []
        for i,t in enumerate(ts):
            print()
            print(i, t.min(), t.max(), len(t))
            f = self.make_f(t)

            # create NAFF object for this time array
            naff = gd.NAFF(t, p=self.p)

            # try recovering the strongest frequency
            w = naff.frequency(f)
            dws.append(np.abs(self.omega[0] - w))
        dws = np.array(dws)
        tt = np.array([t.min() for t in ts])

        from scipy.signal import argrelmax
        ixes = argrelmax(dws)[0]

        print(tt[ixes[1:]] - tt[ixes[:-1]])
        plt.clf()
        plt.semilogy(tt, dws, marker='o', c='k')
        plt.savefig(os.path.join(plot_path, "{0}_rolling_window.png".format(self.__class__.__name__)))

    def test_simple(self):
        ts = [np.linspace(0., 150., 12000),
              np.linspace(0., 150., 24414),
              np.linspace(0., 150., 42104),
              np.linspace(150., 300., 12000),
              np.linspace(150., 300., 24414),
              np.linspace(150., 300., 42104),
              np.linspace(0., 150., 12000) + 50*(2*np.pi/self.omega[0])]

        for i,t in enumerate(ts):
            print()
            print(i, t.min(), t.max(), len(t))
            f = self.make_f(t)

            # create NAFF object for this time array
            naff = gd.NAFF(t)

            # try recovering the strongest frequency
            w = naff.frequency(f)
            print(self.omega[0] - w)
            continue
            # derp

            np.testing.assert_allclose(true_ws[0], w, atol=1E-6)

            # try recovering all frequencies
            output = naff.frecoder(f[:naff.n], break_condition=1E-5)
            nu,A,phi = output
            np.testing.assert_allclose(true_ws, nu[:2], atol=1E-7)
            np.testing.assert_allclose(true_A, A[:2], atol=1E-4)
            np.testing.assert_allclose(true_phi, phi[:2], atol=1E-4)

class TestSimple1(SimpleBase):

    def setup(self):
        self.omega = 2*np.pi*np.array([0.581, 0.73])
        self.amp = np.array([5*(np.cos(np.radians(15.)) + 1j*np.sin(np.radians(15.))),
                             1.8*(np.cos(np.radians(85.)) + 1j*np.sin(np.radians(85.)))])
        self.p = 2

class TestSimple2(SimpleBase):

    def setup(self):
        self.omega = 2*np.pi*np.array([0.581, -0.73, 0.91])
        self.amp = np.array([5*(np.cos(np.radians(35.)) + 1j*np.sin(np.radians(35.))),
                             1.8*(np.cos(np.radians(75.)) + 1j*np.sin(np.radians(75.))),
                             0.7*(np.cos(np.radians(45.)) + 1j*np.sin(np.radians(45.)))])
        self.p = 2

# ----------------------------------------------------------------------------

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
        _d = d[d['idx'] == i]
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

# ----------------------------------------------------------------------------

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

        # read fundamental frequencies
        fn = os.path.join(this_path, "ffs.txt")
        self.fund_freqs = np.loadtxt(fn)

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

    def test_naff_fund_freqs(self):
        naff = NAFF(self.t, p=self.p)

        # complex time series
        if self.poincare:
            logger.info("Using Poincaré polar coordinates")
            fs = poincare_polar(self.w[:,0])
        else:
            logger.info("Using Cartesian coordinates")
            fs = [(self.w[:,0,i] + 1j*self.w[:,0,i+3]) for i in range(3)]

        f,d,ixes = naff.find_fundamental_frequencies(fs[:2], nintvec=15, break_condition=None)

        print("frac. freq diff", (f - self.fund_freqs)/self.fund_freqs)

    @pytest.mark.xfail
    def test_naff_tables(self):
        naff = NAFF(self.t, p=self.p)

        # complex time series
        if self.poincare:
            logger.info("Using Poincaré polar coordinates")
            fs = poincare_polar(self.w[:,0])
        else:
            logger.info("Using Cartesian coordinates")
            fs = [(self.w[:,0,i] + 1j*self.w[:,0,i+3]) for i in range(3)]

        f,d,ixes = naff.find_fundamental_frequencies(fs[:2], nintvec=15, break_condition=None)

        # TEST
        this_d = d[d['idx'] == 0]

        print(this_d['freq'])
        print(self.true_tables[0]['freq'])
        print((this_d['freq'] - self.true_tables[0]['freq']) / self.true_tables[0]['freq'])

        print(this_d['|A|'])
        print(self.true_tables[0]['A'])
        print((this_d['|A|'] - self.true_tables[0]['A']) / self.true_tables[0]['A'])

        phi = coord.Angle(this_d['phi']*u.radian).wrap_at(360*u.deg).to(u.degree).value
        print(phi)
        print(self.true_tables[0]['phi'])
        print((phi - self.true_tables[0]['phi']) / self.true_tables[0]['phi'])

        return

        # fprime = this_d['A'][:,np.newaxis] * np.exp(1j * this_d['freq'][:,np.newaxis] * self.t[np.newaxis])
        # fprime = np.sum(fprime, axis=0)

        # compare with true tables
        for i in range(2):
            # if sum(d['n'] == i) == 0 or i == len(self.true_tables):
            #     break

            this_d = d[d['idx'] == i]
            this_true_tbl = self.true_tables[i][:len(this_d)]
            phi = coord.Angle(this_d['phi']*u.deg).wrap_at(360*u.deg).value

            dfreq = np.abs((this_d['freq'] - this_true_tbl['freq']) / this_true_tbl['freq'])
            dA = np.abs((this_d['|A|'] - this_true_tbl['A']) / this_true_tbl['A'])
            dphi = (phi - this_true_tbl['phi']) / this_true_tbl['phi']

            np.testing.assert_allclose(dfreq[:5], 0., atol=1E-3)
            np.testing.assert_allclose(dA[:5], 0., atol=1E-3)
            np.testing.assert_allclose(dphi[:5], 0., atol=1E-1)

class TestLaskarBox1(LaskarBase):

    def setup_class(self):
        self.name = 'box-orbit-1'
        self.dt = 0.005
        self.nsteps = 2**16
        self.p = 2

class TestLaskarLoop1xy(LaskarBase):

    def setup_class(self):
        self.name = 'loop-orbit-1-xy'
        self.dt = 0.01
        self.nsteps = 2**16
        self.p = 2

class TestLaskarLoop2xy(LaskarBase):

    def setup_class(self):
        self.name = 'loop-orbit-2-xy'
        self.dt = 0.01
        self.nsteps = 2**16
        self.p = 2

class TestLaskarLoop2rphi(LaskarBase):

    def setup_class(self):
        self.name = 'loop-orbit-2-rphi'
        self.dt = 0.01
        self.nsteps = 2**16
        self.p = 2

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
        freqs['idx'] = []
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
            freqs['idx'].append(int(line.split()[4]))
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
            this_d = d[d['idx'] == nq]
            nvec = nvecs[d['idx'] == nq]
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

def test_freq_accuracy_regular():
    potential = gp.LogarithmicPotential(v_c=np.sqrt(2), r_h=0.1,
                                        q1=1., q2=0.9, q3=1., units=galactic)

    nsteps = 100000
    t,w = potential.integrate_orbit([0.49, 0., 0., 1.3156, 0.4788, 0.],
                                    dt=0.004, nsteps=nsteps,
                                    Integrator=DOPRI853Integrator)

    fs = [(w[:nsteps//2,0,i] + 1j*w[:nsteps//2,0,i+3]) for i in range(2)]
    naff = gd.NAFF(t[:nsteps//2], p=2)
    freq1,d,ixes = naff.find_fundamental_frequencies(fs, nintvec=5)

    fs = [(w[nsteps//2:,0,i] + 1j*w[nsteps//2:,0,i+3]) for i in range(2)]
    naff = gd.NAFF(t[:nsteps//2], p=2)
    freq2,d,ixes = naff.find_fundamental_frequencies(fs, nintvec=5)

    print(freq1)
    print(freq2)
    print(freq2 - freq1)

    np.testing.assert_allclose(freq2 - freq1, 0., atol=1E-7)

def test_freq_accuracy_chaotic():
    potential = gp.LogarithmicPotential(v_c=np.sqrt(2), r_h=0.1,
                                        q1=1., q2=0.9, q3=1., units=galactic)

    nsteps = 100000

    # see figure 1 from Papaphillipou & Laskar
    x0 = -0.01
    X0 = -0.2
    y0 = 0.
    E0 = -0.4059
    Y0 = np.sqrt(E0 - potential.value([x0,y0,0.]))
    w0 = np.array([x0,y0,0.,X0,Y0,0.])

    t,w = potential.integrate_orbit(w0, dt=0.004, nsteps=nsteps,
                                    Integrator=DOPRI853Integrator)

    fs = [(w[:nsteps//2,0,i] + 1j*w[:nsteps//2,0,i+3]) for i in range(2)]
    naff = gd.NAFF(t[:nsteps//2], p=4)
    freq1,d,ixes = naff.find_fundamental_frequencies(fs, nintvec=5)

    fs = [(w[nsteps//2:,0,i] + 1j*w[nsteps//2:,0,i+3]) for i in range(2)]
    naff = gd.NAFF(t[:nsteps//2], p=4)
    freq2,d,ixes = naff.find_fundamental_frequencies(fs, nintvec=5)

    print(freq1)
    print(freq2)
    print(freq2 - freq1)

    # assert np.abs(freq2 - freq1).min() > 1E-4

# ------------------------------------------------------------------------------

def roll_it(w0, period, potential, dt=0.005, p=2):
    from ...util import rolling_window

    # total integration time
    nsteps = int(500*period/dt)
    window_size = int(100*period/dt)

    t,w = potential.integrate_orbit(w0, dt=dt, nsteps=nsteps,
                                    Integrator=DOPRI853Integrator)

    # do the rolling window test for the input orbit
    E = potential.total_energy(w[:,0,:3], w[:,0,3:])
    assert np.abs(E[1:] - E[0]).max() < 1E-8

    f = w[:,0,0] + 1j*w[:,0,3]
    naff = gd.NAFF(t, p=p)
    omega0_x = naff.frequency(f)

    f = w[:,0,1] + 1j*w[:,0,4]
    omega0_y = naff.frequency(f)

    # roll the window
    naff = gd.NAFF(t[:window_size], p=p)
    all_freqs = []
    for (ix1,ix2),ww in rolling_window(w[:,0], window_size=window_size, stride=window_size//10, return_idx=True):
        print(ix1, ix2)
        f = ww[:,0] + 1j*ww[:,3]
        freqx = naff.frequency(f, omega0=omega0_x)

        f = ww[:,1] + 1j*ww[:,4]
        freqy = naff.frequency(f, omega0=omega0_y)

        all_freqs.append([freqx, freqy])

    return np.array(all_freqs)

def test_rolling_window_papa():
    # orbits from from Papaphillipou & Laskar
    print()

    potential = gp.LogarithmicPotential(v_c=np.sqrt(2), r_h=0.1,
                                        q1=1., q2=0.9, q3=1., units=galactic)

    # regular orbit:
    w0 = [0.49, 0., 0., 1.3156, 0.4788, 0.]
    print("Regular orbit")
    reg_all_freqs = roll_it(w0, period=2.90454, potential=potential, dt=0.005)
    reg_all_freqs = np.abs(reg_all_freqs[:-1])

    # chaotic orbit initial conditions -- see figure 1 from Papaphillipou & Laskar
    x0 = -0.01
    X0 = -0.2
    y0 = 0.
    E0 = -0.4059
    Y0 = np.sqrt(E0 - potential.value([x0,y0,0.]))
    w0 = np.array([x0,y0,0.,X0,Y0,0.])
    print("Chaotic orbit")
    cha_all_freqs = roll_it(w0, period=1.05, potential=potential)
    cha_all_freqs = np.abs(cha_all_freqs[:-1])

    fig,axes = plt.subplots(1,2,figsize=(12,6))
    # plt.semilogy(np.abs(reg_all_freqs[1:] - reg_all_freqs[0]))
    axes[0].plot(reg_all_freqs[:,0], reg_all_freqs[:,1])
    # plt.semilogy(np.abs(cha_all_freqs[1:] - cha_all_freqs[0]))
    axes[1].plot(cha_all_freqs[:,0], cha_all_freqs[:,1])

    # derivative
    fig,axes = plt.subplots(1,2,figsize=(12,6),sharey=True)

    df = np.sqrt((reg_all_freqs[1:,0] - reg_all_freqs[:-1,0])**2 +
                 (reg_all_freqs[1:,1] - reg_all_freqs[:-1,1])**2)
    # axes[0].semilogy(np.abs(reg_all_freqs[1:,0] - reg_all_freqs[0,0]))
    # axes[0].semilogy(np.abs(reg_all_freqs[1:,1] - reg_all_freqs[0,1]))
    axes[0].semilogy(df)

    df = np.sqrt((cha_all_freqs[1:,0] - cha_all_freqs[:-1,0])**2 +
                 (cha_all_freqs[1:,1] - cha_all_freqs[:-1,1])**2)
    # axes[1].semilogy(np.abs(cha_all_freqs[1:,0] - cha_all_freqs[0,0]))
    # axes[1].semilogy(np.abs(cha_all_freqs[1:,1] - cha_all_freqs[0,1]))
    axes[1].semilogy(df)
    axes[0].set_ylim(1E-9, 1E-2)

    plt.show()

def test_rolling_window_apw():
    # two orbits
    print()

    logger.setLevel(logging.ERROR)

    params = {'a': 1.0, 'b': 0.77, 'c': 0.55, 'r_s': 20.0, 'v_c': 0.17897462888439886}
    potential = gp.LeeSutoTriaxialNFWPotential(units=galactic, **params)

    p = 2

    # regular orbit:
    w0 = [22.76, 0.0, 18.8, 0.0, 0.15610748624460613, 0.0]
    print("Regular orbit")
    reg_all_freqs = roll_it(w0, period=900., potential=potential, dt=2., p=p)
    reg_all_freqs = np.abs(reg_all_freqs[:-1])
    print(reg_all_freqs[1:] - reg_all_freqs[0])

    # chaotic orbit initial conditions
    w0 = [22.76, 0.0, 19.680000000000003, 0.0, 0.15086768887859237, 0.0]
    print("Chaotic orbit")
    cha_all_freqs = roll_it(w0, period=900, potential=potential, dt=2., p=p)
    cha_all_freqs = np.abs(cha_all_freqs[:-1])
    print(cha_all_freqs[1:] - cha_all_freqs[0])

    fig,axes = plt.subplots(1,2,figsize=(12,6))
    axes[0].plot(reg_all_freqs[:,0], reg_all_freqs[:,1])
    axes[1].plot(cha_all_freqs[:,0], cha_all_freqs[:,1])

    # derivative
    fig,axes = plt.subplots(1,2,figsize=(12,6),sharey=True)

    df = np.sqrt((reg_all_freqs[1:,0] - reg_all_freqs[:-1,0])**2 +
                 (reg_all_freqs[1:,1] - reg_all_freqs[:-1,1])**2)
    # axes[0].semilogy(np.abs(reg_all_freqs[1:,0] - reg_all_freqs[0,0]))
    # axes[0].semilogy(np.abs(reg_all_freqs[1:,1] - reg_all_freqs[0,1]))
    axes[0].semilogy(df)

    df = np.sqrt((cha_all_freqs[1:,0] - cha_all_freqs[:-1,0])**2 +
                 (cha_all_freqs[1:,1] - cha_all_freqs[:-1,1])**2)
    # axes[1].semilogy(np.abs(cha_all_freqs[1:,0] - cha_all_freqs[0,0]))
    # axes[1].semilogy(np.abs(cha_all_freqs[1:,1] - cha_all_freqs[0,1]))
    axes[1].semilogy(df)
    axes[0].set_ylim(1E-9, 1E-2)

    plt.show()

# def test_weird_bump():
#     t = np.load(os.path.join(test_data_path, "naff/t.npy"))
#     w = np.load(os.path.join(test_data_path, "naff/w.npy"))

#     every = 2

#     # i1,i2 = (0,200001)
#     # naff = NAFF(t[i1:i2:every], debug=True, debug_path=os.path.join(test_data_path, "naff"))
#     # fs = poincare_polar(w[i1:i2:every])
#     # f,d,ixes = naff.find_fundamental_frequencies(fs, nintvec=5)
#     # print(f)

#     # freqs,d,ixes,is_tube = gd.naff.orbit_to_freqs(t[i1:i2:every], w[i1:i2:every],
#     #                                               silently_fail=False, nintvec=5)
#     # print(freqs)
#     # print()

#     # ----

#     i1,i2 = (350000,550001)
#     naff = NAFF(t[i1:i2:every])
#     fs = poincare_polar(w[i1:i2:every])
#     f,d,ixes = naff.find_fundamental_frequencies(fs[2:3], nintvec=5)
#     print(f)

#     freqs,d,ixes,is_tube = gd.naff.orbit_to_freqs(t[i1:i2:every], w[i1:i2:every],
#                                                   silently_fail=False, nintvec=5)
#     print(freqs)
