# coding: utf-8

""" Port of NAFF to Python """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import logging
import os
import sys
import time

# Third-party
from astropy.constants import G
from astropy import log as logger
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq
from scipy.optimize import fmin_slsqp
from scipy.integrate import simps

# Project

def hanning(x):
    return 1 + np.cos(x)

class NAFF(object):

    def __init__(self, t):
        """ """

        self.t = t
        self.ts = 0.5*(t[-1] + t[0])
        self.T = 0.5*(t[-1] - t[0])
        self.tz = t - self.ts

        # pre-compute values of Hanning filter
        self.chi = hanning(self.tz*np.pi/self.T)

    def frequency(self, f):
        """ Find the most significant frequency of a (complex) time series, `f(t)`,
            by Fourier transforming the function convolved with a Hanning filter and
            picking the biggest peak.
        """

        ndata = len(f)
        C = np.pi / self.T  # normalization constant to go from [-T/2,T/2] to [-pi,pi]

        # take Fourier transform of input (complex) function f
        logger.debug("Fourier transforming...")
        t1 = time.time()
        fff = fft(f) / np.sqrt(ndata)
        omegas = 2*np.pi*fftfreq(f.size, self.t[1]-self.t[0])
        logger.debug("...done. Took {} seconds to FFT.".format(time.time()-t1))

        A = 1./np.sqrt(ndata - 1.)
        xf = A * fff.real * (-1)**np.arange(ndata)
        yf = A * fff.imag * (-1)**np.arange(ndata)

        # find max of xf, yf, get index of row -- this is just an initial guess
        xyf = np.vstack((xf,yf))
        xyf_abs = np.abs(xyf)
        wmax = np.max(xyf_abs, axis=0).argmax()
        if xf[wmax] != 0.:
            signx = np.sign(xf[wmax])
            signy = np.sign(xf[wmax])
            logger.debug("sign(xf) = {}".format(signx))
            logger.debug("sign(yf) = {}".format(signy))
            # signx = np.sign(xyf[xyf_abs[:,wmax].argmax(),wmax])
        else:
            # return early -- "this may be an axial or planar orbit"
            return 0.

        # find the frequency associated with this index
        omega0 = omegas[wmax]
        signo = np.sign(omega0)
        logger.debug("sign(omega0) = {}".format(signo))

        # now that we have a guess for the maximum, convolve with Hanning filter and re-solve
        xf = f.real
        yf = f.imag

        # window around estimated best frequency
        omin = omega0 - np.pi/self.T
        omax = omega0 + np.pi/self.T

        def phi_w(w):
            """ This function numerically computes phi(ω), as in
                Eq. 12 in Valluri & Merritt (1998).
            """

            # real part of integrand of Eq. 12
            zreal = self.chi * (xf*np.cos(w*self.tz) + yf*np.sin(w*self.tz))
            ans = simps(zreal, x=self.tz)
            return -(ans*signx*signo)/(2.*self.T)

        # ----------------------------------------------------------------
        # figure out if concave up or down
        # vals = np.linspace(omin,omax,100)
        # derp = np.zeros_like(vals)
        # for i in range(1,len(vals)-1):
        #     derp[i] = phi_w(vals[i-1]) + -2*phi_w(vals[i]) + phi_w(vals[i+1])

        # if np.median(derp) < 0.:
        #     print("concave down")
        #     plt.title("Concave down")
        # else:
        #     print("concave up")
        #     plt.title("Concave up")

        # # plot
        # vals = np.linspace(omin,omax,100)
        # derp = np.zeros_like(vals)
        # for i,val in enumerate(vals):
        #     derp[i] = phi_w(val)
        # plt.plot(vals, derp)
        # plt.show()
        # sys.exit(0)
        # ----------------------------------------------------------------

        res = fmin_slsqp(phi_w, x0=(omax+omin)/2, acc=1E-12,
                         bounds=[(omin,omax)], disp=0, iter=50,
                         full_output=True)

        freq,fx,its,imode,smode = res
        if imode != 0:
            raise ValueError("Function minimization to find best frequency failed with:\n"
                             "\t {} : {}".format(imode, smode))

        return abs(freq[0])

    def frecoder(self, t, f, nvec=12):
        """ Same as the subroutine FRECODER in Valluri's NAFF routines. """

        # initialize container arrays
        ecap = np.zeros((nvec,len(t)), dtype=np.complex64)
        nu = np.zeros(nvec)
        ai = np.zeros(nvec, dtype=np.complex64)
        # phi = np.zeros(nvec)

        # put this in loop below if we do need phi:
        # if ai[k].real != 0.:
        #     phi[k] = np.arctan(ai[k].imag) / ai[k].real
        # else:
        #     phi[k] = 0.5*np.pi

        fk = f.copy()
        for k in range(nvec):
            nu[k] = self.frequency(fk)
            logger.info("frequency ω = {} for k = {}".format(nu[k], k))

            if k == 0:
                # compute exp(iωt) for first frequency
                ecap[k] = np.cos(nu[k]*self.tz) + 1j*np.sin(nu[k]*self.tz)

                # get complex amplitude by projecting exp(iωt) on to f(t)
                ai[k] = self.hanning_product(f, ecap[k])

                # new fk has the previous frequency subtracted out
                fk,fmax = self.sub_chi(fk, ecap[k], ai[k])
                continue

            ecap[k] = self.gso(ecap, nu[k], k)
            ai[k] = self.hanning_product(fk, ecap[k])
            fk,fmax = self.sub_chi(fk, ecap[k], ai[k])

            # TODO: why?
            if fmax < 1E-7 or np.abs(ai[k]) < 1E-7:
                break

        return nu[:k+1], ai[:k+1]

    def hanning_product(self, u1, u2):
        # Routine to compute the scalar product of two vectors
        # the scalar product is defined with the Hanning filter as
        #  <u1, u2> = integral(u1(t)* chi(t) * u2~(t))
        # The product is the integral 1/2t*(u1(t)*chi(t)*u2conj(t) dt)

        # First find complex conjugate of vector u2 and construct integrand
        integ = u1 * np.conj(u2) * self.chi
        integ_r = integ.real
        integ_i = integ.imag

        # Now integrate the real part:
        real = simps(integ_r, x=self.tz) / (2.*self.T)

        # Integrate Imaginary part:
        imag = simps(integ_i, x=self.tz) / (2.*self.T)

        return real + imag*1j

    def gso(self, ecap, nu, k):
        """ Gram-Schmidt Orthogonalization of

        Parameters
        ----------
        ecap : ndarray
        nu : numeric
        k : int
            Index of maximum freq. found so far.
        """
        cik = np.zeros(k, dtype=np.complex64)

        u_n = np.cos(nu*self.tz) + 1j*np.sin(nu*self.tz)

        # first find the k complex constants cik(k,ndata):
        for j in range(k):
            cik[j] = self.hanning_product(u_n, ecap[j])

        # Now construct the orthogonal vector
        # print(np.sum(cik[:,np.newaxis]*ecap[:k], axis=0))
        e_i = u_n - np.sum(cik[:,np.newaxis]*ecap[:k], axis=0)

        # Now Normalize this vector:
        # <ei, ei> = A + iB
        prod = self.hanning_product(e_i, e_i)

        if prod != 0.:
            norm = 1. / np.sqrt(prod)
        else:
            norm = 0. + 0j

        e_i *= norm

        # now fill in the (k)th vector into the ecap array
        return e_i

    def sub_chi(self, f_km1, ecap_k, a_k):
        # remove the new orthogonal frequency component from the f_k
        f_k = f_km1 - a_k*ecap_k

        # now compute the largest amplitude of the residual function f_k
        fmax = np.max(np.abs(f_k))

        return f_k, fmax

def main(t, fs):
    naff = NAFF(t)

    for f in fs:
        nu_f,a_f = naff.frecoder(t, f, nvec=5)

    sys.exit(0)

    fq = np.abs(np.concatenate((nux, nuy, nuz)))
    ampc = np.concatenate((aix, aiy, aiz))
    amp = np.abs(ampc)
    k = np.concatenate((range(nux.size),
                        range(nuy.size),
                        range(nuz.size)))
    nq = np.concatenate((np.zeros_like(nux),
                         np.ones_like(nuy),
                         np.ones_like(nuz)+1.)).astype(int)

    ktot = nux.shape[0] + nuy.shape[0] + nuz.shape[0]
    ix = amp.argsort()[::-1]

    fqs = fq[ix]
    nq = nq[ix]
    amp = amp[ix]
    ampc = ampc[ix]

    print(nux)
    print(nuy)
    print(nuz)
    sys.exit(0)
    print(fqs[:10])

    f = np.zeros(3)
    k3 = np.zeros(3)
    ss = np.zeros(3, dtype=np.complex64)

    f[0] = fqs[0]
    k3[0] = nq[0]
    ss[0] = ampc[0]

    l1 = 0
    # choose the next nontrivially related frequency as the 2nd fundamental:
    for i in range(1,ktot):
        if np.abs(np.abs(f[0]) - np.abs(fqs[i])) > 1E-6:
            print(k3[0], nq[i])
            if k3[0] != nq[i]:
                f[1] = fqs[i]
                k3[1] = nq[i]
                ss[1] = ampc[i]
                l2 = i
                break

    # brute-force method for finding third frequency
    n1 = np.zeros_like(fqs)
    n2 = np.zeros_like(fqs)
    n3 = np.zeros_like(fqs)
    err = np.zeros_like(fqs)

    imax = 15

    for i in range(fqs.shape[0]):
        obji = 1E20
        for in1 in range(-imax,imax+1,1):
            for in2 in range(-imax,imax+1,1):
                #for in3 in range(-imax,imax+1,1):
                funi = np.abs(fqs[i] - in1*f[0] - in2*f[1])  #  - in3*f[2])
                if funi < obji:
                    obji = funi
                    n1[i] = in1
                    n2[i] = in2
                    err[i] = obji

    f[2] = fqs[err.argmax()]

    print(f)

    # ----------------------------------- (IP stuff)
    # # now, for each frequency in the list, try until the first unsolvable
    # #   frequency is found -- this is the third frequency component
    # cvec = np.zeros([f[0], f[1], 0., np.nan])

    # A = np.zeros((1,4))
    # for i in range(l2,ktot):
    #     A[0,3] = -f[i]
    #     b = np.array([1.])

if __name__ == '__main__':
    from argparse import ArgumentParser
    import gary.coordinates as gc
    import gary.dynamics as gd
    import gary.potential as gp
    import gary.integrate as gi
    from gary.units import galactic

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")

    # parser.add_argument("-f", dest="field_id", default=None, required=True,
    #                     type=int, help="Field ID")
    # parser.add_argument("-p", dest="plot", action="store_true", default=False,
    #                     help="Plot or not")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    # -----------------------------------------------------------------------
    # Hand-constructed time-series
    # Ts = (1., 1.2, 1.105)
    # As = (1., 0.5, 0.2)
    # # Ts = (1., )
    # # As = (1., )
    # t = np.linspace(0,100,50000)

    # f = np.sum([A*(np.cos(2*np.pi*t/T) + 1j*np.sin(2*np.pi*t/T)) for T,A in zip(Ts,As)], axis=0)
    # fs = (f,)

    # print("True freqs:")
    # for T in Ts:
    #     print(2*np.pi/T)

    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Easy:
    # Ts = np.array([1., 1.2, 1.105])
    # freqs = 2*np.pi/Ts
    # print("True freqs:", freqs)
    # t2 = 100
    # nsteps = 50000
    # dt = t2/float(nsteps)

    # fn = "orbit.npy"
    # if not os.path.exists(fn):
    #     logger.debug("Integrating orbit...")
    #     potential = gp.HarmonicOscillatorPotential(freqs)
    #     t,w = potential.integrate_orbit([1,0,0.2,0.,0.1,-0.8], dt=dt, nsteps=nsteps,
    #                                     Integrator=gi.DOPRI853Integrator)
    #     np.save(fn, np.vstack((w.T,t[np.newaxis,np.newaxis])))
    # wt = np.load(fn)
    # w = wt[:6].T.copy()
    # t = wt[6,0].copy()

    # fs = (w[:,0,0] + 1j*w[:,0,3],
    #       w[:,0,1] + 1j*w[:,0,4],
    #       w[:,0,2] + 1j*w[:,0,5])

    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Logarithmic potential, 2D orbit as in Papaphilippou & Laskar (1996):
    #   Table 1
    fn = "orbit_log.npy"

    if not os.path.exists(fn):
        logger.debug("Integrating orbit...")
        potential = gp.LogarithmicPotential(v_c=np.sqrt(2.), r_h=0.1,
                                            q1=1., q2=0.9, q3=1., units=galactic)
        t,w = potential.integrate_orbit([0.49,0.,0.,1.3156,0.4788,0.], dt=0.005, nsteps=50000,
                                        Integrator=gi.DOPRI853Integrator)
        np.save(fn, np.vstack((w.T,t[np.newaxis,np.newaxis])))
    wt = np.load(fn)
    w = wt[:6].T.copy()
    t = wt[6,0].copy()

    fs = (w[:,0,0] + 1j*w[:,0,3],
          w[:,0,1] + 1j*w[:,0,4])

    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Logarithmic potential, 1D orbit as in Papaphilippou & Laskar (1996):
    #   Table 2
    # fn = "orbit_log1d.npy"

    # if not os.path.exists(fn):
    #     logger.debug("Integrating orbit...")
    #     potential = gp.LogarithmicPotential(v_c=np.sqrt(2.), r_h=0.1,
    #                                         q1=1., q2=0.9, q3=1., units=galactic)
    #     t,w = potential.integrate_orbit([0.49,0.,0.,1.4,0.,0.], dt=0.005, nsteps=50000,
    #                                     Integrator=gi.DOPRI853Integrator)
    #     np.save(fn, np.vstack((w.T,t[np.newaxis,np.newaxis])))
    # wt = np.load(fn)
    # w = wt[:6].T.copy()
    # t = wt[6,0].copy()

    # fs = (w[:,0,0] + 1j*w[:,0,3], )

    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Harder? For loop orbit, align z axis with circulation?
    # fn = "orbit2.npy"
    # potential = gp.IsochronePotential(m=1E11, b=1., units=galactic)
    # if not os.path.exists(fn):
    #     logger.debug("Integrating orbit...")
    #     t,w = potential.integrate_orbit([1.,0,0.1,0.,0.08,0.06], dt=0.05, nsteps=2**14,
    #                                     Integrator=gi.DOPRI853Integrator)
    #     np.save(fn, np.vstack((w.T,t[np.newaxis,np.newaxis])))
    # wt = np.load(fn)
    # w = wt[:6].T.copy()
    # t = wt[6,0].copy()

    # fs = (w[:,0,0] + 1j*w[:,0,3],
    #       w[:,0,1] + 1j*w[:,0,4],
    #       w[:,0,2] + 1j*w[:,0,5])

    # # Compute true frequencies
    # G,m,b = G.decompose(galactic).value, potential.parameters['m'], potential.parameters['b']
    # E = potential.total_energy(w[:,0,:3],w[:,0,3:]).mean()
    # L = np.linalg.norm(np.cross(w[:,0,:3],w[:,0,3:]), axis=-1).mean()

    # Jr = G*m/np.sqrt(-2*E) - 0.5*(L + np.sqrt(L**2 + 4*G*m*b))
    # Jr = Jr.mean()

    # freq_r = np.mean((G*m)**2 / (Jr + 0.5*(L + np.sqrt(L**2 + 4*G*m*b)))**3)
    # freq_phi = np.mean(0.5*(1 + L/np.sqrt(L**2 + 4*G*m*b))*freq_r)

    # print("freq (r,φ)", freq_r, freq_phi)
    # print(np.abs(freq_r - freq_phi))

    # compute frequencies with Sanders
    # actions,angles,freqs = gd.find_actions(t, w, N_max=10, units=galactic)
    # print(freqs)
    # sys.exit(0)

    # -----------------------------------------------------------------------

    # # plot orbit
    # fig = gd.plot_orbits(w, marker='.', linestyle='none', alpha=0.2)
    # plt.show()

    # # plot energy conservation
    # E = potential.total_energy(w[:,0,:3],w[:,0,3:])
    # plt.semilogy(t[1:], np.abs(E[1:]-E[:-1]), marker=None)
    # plt.show()

    # sys.exit(0)

    logger.debug("Done integrating orbit, starting frequency analysis...")
    main(t, fs)
