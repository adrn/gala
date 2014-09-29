# coding: utf-8

""" Test helpers """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# Project
from ...units import galactic
from ...coordinates import spherical_to_cartesian

# HACK:
if "/Users/adrian/projects/genfunc" not in sys.path:
    sys.path.append("/Users/adrian/projects/genfunc")
import genfunc_3d

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

def sanders_act_ang_freq(t, w, N_max=6):
    w2 = w[:,0].copy()
    w2[:,3:] = (w2[:,3:]*u.kpc/u.Myr).to(u.km/u.s).value
    act,ang,n_vec,toy_aa,pars = genfunc_3d.find_actions(w2, t/1000., N_matrix=N_max)

    actions = (act[:3]*u.kpc*u.km/u.s).to(u.kpc**2/u.Myr).value
    angles = ang[:3]
    freqs = (ang[3:6]/u.Gyr).to(1/u.Myr).value

    return actions,angles,freqs

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

    fig,axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(10,5))
    _crazy_angle_loop(theta[0], theta[1], axes[0])
    _crazy_angle_loop(theta[0], theta[2], axes[1])

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

    x,v = spherical_to_cartesian(d, phi, theta,
                                 vr.decompose(galactic).value,
                                 vphi.decompose(galactic).value,
                                 vtheta.decompose(galactic).value)

    return np.hstack((x,v))
