# coding: utf-8

""" Port of NAFF to Python """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftshift
from scipy.optimize import fmin_slsqp
from scipy.integrate import trapz

# Project

def phir(w, xf, yf, signx, chi, tz, tg):
    # this is the subroutine phir() in NAFF
    zreal = chi * (xf*np.cos(w*tz) + yf*np.sin(w*tz))
    ans = trapz(zreal, x=tz)
    return -(ans*signx)/(2.*tg)

def hanning(x):
    return 1 + np.cos(x)

def hanning_product(u1, u2, chi, tz, tg):
    # Routine to compute the scalar product of two vectors
    # the scalar product is defined with the Hanning filter as
    #  <u1, u2> = integral(u1(t)* chi(t) * u2~(t))
    # The product is the integral 1/2t*(u1(t)*chi(t)*u2conj(t) dt)

    # First find complex conjugate of vector u2 and construct integrand
    u2conj = np.conj(u2)
    integ = u1 * u2conj * chi
    integ_r = integ.real
    integ_i = integ.imag

    # Now integrate the real part:
    ans1 = trapz(integ_r, x=tz)
    ans1 = ans1/(2.*tg)

    # Integrate Imaginary part:
    ans2 = trapz(integ_i, x=tz)
    ans2 = ans2/(2.*tg)

    return ans1 + ans2*1j

def frequency(f, tz, tg, chi):
    ndat = len(f)
    const = np.pi / tg
    fff = fft(f)

    xf = fff.real * (-1)**np.arange(0,ndat,1)/np.sqrt(len(fff) - 1.)
    yf = fff.imag * (-1)**np.arange(0,ndat,1)/np.sqrt(len(fff) - 1.)

    derp = np.max(np.abs(np.vstack((xf,yf))), axis=0)
    # xmax = derp.max()
    wmax = derp.argmax()

    if xf[wmax] != 0.:
        signx = np.sign(xf[wmax])
    else:
        # return early -- "this may be an axial or planar orbit"
        return 0.

    if wmax <= ndat/2.:
        omega0 = const * wmax
    else:
        omega0 = const * (wmax-ndat)

    xf = fff.real
    yf = fff.imag

    omin = omega0 - np.pi/tg
    omax = omega0 + np.pi/tg

    res = fmin_slsqp(phir, x0=(omax+omin)/2, args=(xf, yf, signx, chi, tz, tg),
                     acc=1E-15, bounds=[(omin,omax)])

    # freqi = abs(res.x)[0]
    # ampli = -res.fun
    freqi = abs(res[0])
    # ampli = res[1]

    return freqi

def sub_chi(f_km1, k, ecap, ai):
    # remove the new orthogonal frequency component from the f_k
    f_k = f_km1 - ai[k]*ecap[k]

    # now compute the mean amplitude of the residual function f_k
    chik = np.cumsum(np.abs(f_k**2))
    fmax = np.max(f_k)
    chik = np.sqrt(chik / len(f_km1))

    return f_k, fmax

def gso(ecap, nui, k, chi, tz, tg):
    cik = np.zeros(k, dtype=np.complex64)

    # generate the u_n vector:
    ux = np.cos(nui*tz)
    uy = np.sin(nui*tz)
    u_n = ux + uy*1j

    # on input k = number of vectors found so far:
    # first find the k-1 complex constants cik(k,2):

    for j in range(k):
        e_i = ecap[j]
        cik[j] = hanning_product(u_n, e_i, chi, tz, tg)

    # Now construct the orthogonal vector
    for i in range(len(tz)):
        sum = np.sum(cik[j] * ecap[j])
        e_i[i] = u_n[i] - sum

    # Now Normalize this vector:
    # <ei, ei> = A +iB,  e^i = ei/sqrt(A+iB)
    prod = hanning_product(e_i, e_i, chi, tz, tg)

    if prod != 0.:
        norm = 1. / np.sqrt(prod)
    else:
        norm = 0. + 0j

    e_i *= norm

    # now fill in the (k)th vector into the ecap array
    ecap[k] = e_i

def frecoder(t, f, nvec=10):
    ecap = np.zeros((nvec,len(t)), dtype=np.complex64)
    nu_k = np.zeros(nvec)
    ai = np.zeros(nvec, dtype=np.complex64)
    amp = np.zeros(nvec)
    phi = np.zeros(nvec)

    ts = 0.5*(t[-1] + t[0])
    tg = 0.5*(t[-1] - t[0])
    tz = t - ts
    const = np.pi / tg

    chi = hanning(tz*const)
    fk = f.copy()
    for k in range(nvec):
        # TODO: break if tolerance met for .. something
        nu_k[k] = frequency(fk, tz, tg, chi)

        if k == 0:
            xx = np.cos(nu_k[0]*tz)
            yy = np.sin(nu_k[0]*tz)
            ecap[0] = xx + yy*1j

            ai[k] = hanning_product(fk, ecap[0], chi, tz, tg)
            amp[k] = np.abs(ai[k])

            if ai[k].real != 0.:
                phi[k] = np.arctan(ai[k].imag) / ai[k].real
            else:
                phi[k] = 0.5*np.pi

            fk,fmax = sub_chi(fk, k, ecap, ai)
            continue

        gso(ecap, nu_k[k], k, chi, tz, tg)  # modifies ecap in place
        ai[k] = hanning_product(fk, ecap[k], chi, tz, tg)
        amp[k] = np.abs(ai[k])

        if ai[k].real != 0.:
            phi[k] = np.arctan(ai[k].imag) / ai[k].real
        else:
            phi[k] = 0.5*np.pi

        fk,fmax = sub_chi(fk, k, ecap, ai)

        if fmax < 1E-8 or np.abs(ai[k]) < 1E-8:
            break

    return nu_k[:k+1], ai[:k+1]

def main(t, w):
    fx = w[:,0,0] + w[:,0,3] * 1j
    fy = w[:,0,1] + w[:,0,4] * 1j
    fz = w[:,0,2] + w[:,0,5] * 1j

    nux,aix = frecoder(t, fx)
    print(nux, aix)

    nuy,aiy = frecoder(t, fy)
    print(nuy, aiy)

    nuz,aiz = frecoder(t, fz)
    print(nuz, aiz)

    nu = np.append(np.append(nux, nuy), nuz)
    ai = np.append(np.append(aix, aiy), aiz)

    ix = np.abs(ai).argsort()[::-1]

    print()
    print("Best freq:", nu[ix][0])
    print("Top 10 freqs:", nu[ix][:10])


if __name__ == '__main__':
    import gary.potential as gp
    import gary.integrate as gi

    potential = gp.HarmonicOscillatorPotential([1.214, 1.46, 1.1])
    t,w = potential.integrate_orbit([1,0,0.2,0.,0.1,-0.8], dt=0.1, nsteps=10000,
                                    Integrator=gi.DOPRI853Integrator)

    # E = potential.total_energy(w[:,0,:3],w[:,0,3:])
    # plt.semilogy(t[1:], np.abs(E[1:]-E[:-1]), marker=None)
    # plt.show()

    # TODO: need a better integrator mehbeh?

    main(t,w)
