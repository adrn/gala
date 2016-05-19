# coding: utf-8

""" Test helpers """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# Project
# from ..actionangle import classify_orbit
from ...units import galactic
from ...coordinates import physicsspherical_to_cartesian
from ...potential import HarmonicOscillatorPotential, IsochronePotential
from .._genfunc import genfunc_3d, solver, toy_potentials

def sanders_nvecs(N_max, dx, dy, dz):
    from itertools import product
    NNx = range(-N_max, N_max+1, dx)
    NNy = range(-N_max, N_max+1, dy)
    NNz = range(-N_max, N_max+1, dz)
    n_vectors = np.array([[i,j,k] for (i,j,k) in product(NNx,NNy,NNz)
                          if(not(i==0 and j==0 and k==0)            # exclude zero vector
                             and (k>0                               # northern hemisphere
                                  or (k==0 and j>0)                 # half of x-y plane
                                  or (k==0 and j==0 and i>0))       # half of x axis
                             and np.sqrt(i*i+j*j+k*k)<=N_max)])     # inside sphere
    return n_vectors

def sanders_act_ang_freq(t, w, circ, N_max=6):
    w2 = w.copy()

    if np.any(circ):
        w2[3:] = (w2[3:]*u.kpc/u.Myr).to(u.km/u.s).value
        (act,ang,n_vec,toy_aa,pars),loop2 = genfunc_3d.find_actions(w2.T, t/1000.,
                                                                    N_matrix=N_max, ifloop=True)
    else:
        (act,ang,n_vec,toy_aa,pars),loop2 = genfunc_3d.find_actions(w2.T, t,
                                                                    N_matrix=N_max, ifloop=True)

    actions = act[:3]
    angles = ang[:3]
    freqs = ang[3:6]

    if np.any(circ):
        toy_potential = IsochronePotential(m=pars[0]*1E11, b=pars[1], units=galactic)
        actions = (actions*u.kpc*u.km/u.s).to(u.kpc**2/u.Myr).value
        freqs = (freqs/u.Gyr).to(1/u.Myr).value
    else:
        toy_potential = HarmonicOscillatorPotential(omega=np.array(pars), units=galactic)

    return actions,angles,freqs,toy_potential

def _crazy_angle_loop(theta1,theta2,ax):
    cnt = 0
    ix1 = 0
    while True:
        cnt += 1

        for ix2 in range(ix1,ix1+1000):
            if ix2 > len(theta1)-1:
                ix2 = len(theta1)-1
                break

            if theta1[ix2] < theta1[ix1] or theta2[ix2] < theta2[ix1]:
                ix2 -= 1
                break

        if theta1[ix2] != theta1[ix1:ix2+1].max() or theta2[ix2] != theta2[ix1:ix2+1].max():
            ix1 = ix2+1
            continue

        if cnt > 100 or ix2 == len(theta1)-1:
            break

        if ix1 == ix2:
            ix1 = ix2+1
            continue

        ax.plot(theta1[ix1:ix2+1], theta2[ix1:ix2+1], alpha=0.5, marker='o', c='k')

        ix1 = ix2+1

def plot_angles(t, angles, freqs, subsample_factor=1000):
    theta = (angles[:,None] + freqs[:,None]*t[np.newaxis])
    subsample = theta.shape[1]//subsample_factor
#    subsample = 1
    theta = (theta[:,::subsample] / np.pi) % 2.
    print(theta.shape)
    fig,axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(10,5))
    # _crazy_angle_loop(theta[0], theta[1], axes[0])
    # _crazy_angle_loop(theta[0], theta[2], axes[1])
    axes[0].plot(theta[0], theta[1], ls='none')
    axes[0].plot(theta[0], theta[2], ls='none')

    axes[0].set_xlim(0,2)
    axes[0].set_ylim(0,2)
    return fig
    # axes[1].scatter(theta[0,ix], theta[2], alpha=0.5, marker='o', c=t)

def isotropic_w0(N=100):
    # positions
    d = np.random.lognormal(mean=np.log(25), sigma=0.5, size=N)
    phi = np.random.uniform(0, 2*np.pi, size=N)
    theta = np.arccos(np.random.uniform(size=N) - 0.5)

    vr = np.random.normal(150., 40., size=N)*u.km/u.s
    vt = np.random.normal(100., 40., size=N)
    vt = np.vstack((vt,np.zeros_like(vt))).T

    # rotate to be random position angle
    pa = np.random.uniform(0, 2*np.pi, size=N)
    M = np.array([[np.cos(pa), -np.sin(pa)],[np.sin(pa), np.cos(pa)]]).T
    vt = np.array([vv.dot(MM) for (vv,MM) in zip(vt,M)])*u.km/u.s
    vphi,vtheta = vt.T

    rep = coord.PhysicsSphericalRepresentation(r=d*u.dimensionless_unscaled,
                                               phi=phi*u.radian,
                                               theta=theta*u.radian)
    x = rep.represent_as(coord.CartesianRepresentation).xyz.T.value

    vr = vr.decompose(galactic).value
    vphi = vphi.decompose(galactic).value
    vtheta = vtheta.decompose(galactic).value
    v = physicsspherical_to_cartesian(rep, [vr,vphi,vtheta]*u.dimensionless_unscaled).T.value

    return np.hstack((x,v)).T
