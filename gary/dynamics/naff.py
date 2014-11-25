# coding: utf-8

""" Port of NAFF to Python """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import sys
import time

# Third-party
from astropy import log as logger
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

        res = fmin_slsqp(phi_w, x0=(omax+omin)/2, acc=1E-12,
                         bounds=[(omin,omax)], disp=0, iter=50,
                         full_output=True)

        freq,fx,its,imode,smode = res
        if imode != 0:
            raise ValueError("Function minimization to find best frequency failed with:\n"
                             "\t {} : {}".format(imode, smode))

        return freq[0]

    def frecoder(self, f, nvec=12):
        """ Same as the subroutine FRECODER in Valluri's NAFF routines. """

        # initialize container arrays
        ecap = np.zeros((nvec,len(t)), dtype=np.complex64)
        nu = np.zeros(nvec)
        A = np.zeros(nvec)
        phi = np.zeros(nvec)

        fk = f.copy()
        logger.info("-"*50)
        logger.info("k    ωk    Ak    φk(deg)    ak")
        for k in range(nvec):
            nu[k] = self.frequency(fk)

            if k == 0:
                # compute exp(iωt) for first frequency
                ecap[k] = np.cos(nu[k]*self.tz) + 1j*np.sin(nu[k]*self.tz)
            else:
                ecap[k] = self.gso(ecap, nu[k], k)

            # get complex amplitude by projecting exp(iωt) on to f(t)
            ab = self.hanning_product(fk, ecap[k])
            A[k] = np.abs(ab)
            phi[k] = np.arctan2(ab.imag, ab.real)

            # new fk has the previous frequency subtracted out
            fk,fmax = self.sub_chi(fk, ecap[k], ab)

            logger.info("{}  {:.6f}  {:.6f}  {:.2f}  {:.6f}"
                        .format(k,nu[k],A[k],np.degrees(phi[k]),ab))

            # TODO: why?
            if fmax < 1E-7 or A[k] < 1E-6:
                break

        return nu[:k+1], A[:k+1], phi[:k+1]

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

    def find_fundamental_frequencies(self, w, nvec=15):
        """ Solve for the fundamental frequencies of the given orbit, `w`.

            TODO:
        """

        if w.ndim > 2:
            raise ValueError("Input orbit must be a single orbit (have ndim=2).")

        # containers
        freqs = []
        As = []
        amps = []
        phis = []
        nqs = []

        ntot = 0
        ndim = w.shape[1]//2
        for i in range(ndim):
            nu,A,phi = self.frecoder(w[:,i] + 1j*w[:,i+ndim], nvec=nvec)
            freqs.append(nu)
            As.append(A)
            amps.append(np.abs(A))
            phis.append(phi)
            nqs.append(np.zeros_like(nu) + i)
            ntot += len(nu)

        d = np.zeros(ntot, dtype=zip(('freq','A','|A|','phi','n'),
                                     ('f8','c8','f8','f8',np.int)))
        d['freq'] = np.concatenate(freqs)
        d['A'] = np.concatenate(As)
        d['|A|'] = np.concatenate(amps)
        d['phi'] = np.concatenate(phis)
        d['n'] = np.concatenate(nqs).astype(int)

        # sort terms by amplitude
        d = d[d['|A|'].argsort()[::-1]]

        # assume largest amplitude is the first fundamental frequency
        ffreq = np.zeros(ndim)
        ffreq_ixes = np.zeros(ndim, dtype=int)
        ffreq[0] = d[0]['freq']

        # choose the next nontrivially related frequency as the 2nd fundamental:
        #   TODO: why 1E-6? this isn't well described in the papers...
        ixes = np.where((d['n'] != d[0]['n']) & ((np.abs(ffreq[0]) - np.abs(d['freq'])) > 1E-6))[0]
        ffreq[1] = d[ixes[1]]['freq']
        ffreq_ixes[1] = ixes[1]

        # brute-force method for finding third frequency: find maximum error in
        #   n*f1 + m*f2 - l*f3
        n1 = np.zeros(ntot)
        n2 = np.zeros(ntot)
        err = np.zeros(ntot)

        imax = 15
        for i in range(ntot):
            obji = 1E20
            for in1 in range(-imax,imax+1,1):
                for in2 in range(-imax,imax+1,1):
                    # for in3 in range(-imax,imax+1,1):
                    funi = np.abs(d[i]['freq'] - in1*ffreq[0] - in2*ffreq[1])
                    if funi < obji:
                        obji = funi
                        n1[i] = in1
                        n2[i] = in2
                        err[i] = obji

        ffreq[2] = d[err.argmax()]['freq']
        ffreq_ixes[2] = err.argmax()

        return ffreq, d, ffreq_ixes
