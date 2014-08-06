# coding: utf-8

""" Produce plots for the Action-Angle tutorial. To run the three examples,
    execute::

        python dynamics.py --name axisymmetricloop
        python dynamics.py --name triaxialloop
        python dynamics.py --name triaxialchaotic

"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np
from astropy import log as logger
import astropy.units as u
import matplotlib.pyplot as plt

# Package
import streamteam.potential as sp
import streamteam.integrate as si
import streamteam.dynamics as sd

logger.setLevel(logging.DEBUG)

plot_path = os.path.abspath("../_static/dynamics")
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

def make_orbit_files(potential, w0, N_max=6, suffix="", overwrite=False,
                     force_harmonic_oscillator=False):

    orbit_filename = os.path.join(plot_path, "orbits{}.npy".format(suffix))
    usys = potential.usys
    nsteps = 50000
    dt = 10.

    #NT = 9*N_max**3/2
    #every = nsteps // NT

    if overwrite and os.path.exists(orbit_filename):
        os.remove(orbit_filename)

    toy_potential = None
    if not os.path.exists(orbit_filename):
        # integrate an orbit in a axisymmetric potential
        logger.debug("Integrating orbit for {} steps...".format(nsteps))
        acc = lambda t,w: np.hstack((w[...,3:],potential.acceleration(w[...,:3])))
        integrator = si.DOPRI853Integrator(acc)
        # acc = lambda t,x: potential.acceleration(x)
        # integrator = si.LeapfrogIntegrator(acc)
        t,w = integrator.run(w0, dt=dt, nsteps=nsteps)
        logger.debug("...done!")

        loop = sd.classify_orbit(w)
        if np.any(loop == 1) and not force_harmonic_oscillator: # loop orbit
            logger.debug("Orbit classified as LOOP")
            m,b = sd.fit_isochrone(w, usys=usys)
            toy_potential = sp.IsochronePotential(m=m, b=b, usys=usys)
        else:
            logger.debug("Orbit classified as BOX")
            omegas = sd.fit_harmonic_oscillator(w, usys=usys)
            toy_potential = sp.HarmonicOscillatorPotential(omega=omegas, usys=usys)

        # also integrate the orbit in the best-fitting toy potential
        toy_steps = w.shape[0]//10
        logger.debug("Integrating toy orbit for {} steps...".format(toy_steps))
        acc = lambda ts,ws: np.hstack((ws[...,3:],toy_potential.acceleration(ws[...,:3])))
        integrator = si.DOPRI853Integrator(acc)
        # acc = lambda t,x: toy_potential.acceleration(x)
        # integrator = si.LeapfrogIntegrator(acc)
        toy_t,toy_w = integrator.run(w0, dt=dt, nsteps=toy_steps)
        logger.debug("...done!")

        # cache the orbits
        np.save(orbit_filename, (t,w,toy_t,toy_w))

        logger.debug("Orbit computed and saved to file: {}".format(orbit_filename))
    else:
        t,w,toy_t,toy_w = np.load(orbit_filename)
        logger.debug("Orbit read from file: {}".format(orbit_filename))

    if toy_potential is None:
        loop = sd.classify_orbit(w)
        if np.any(loop == 1) and not force_harmonic_oscillator: # loop orbit
            m,b = sd.fit_isochrone(w, usys=usys)
            toy_potential = sp.IsochronePotential(m=m, b=b, usys=usys)
        else:
            omegas = sd.fit_harmonic_oscillator(w, usys=usys)
            toy_potential = sp.HarmonicOscillatorPotential(omega=omegas, usys=usys)

    # plot a smaller section of the orbit in projections of XYZ
    plot_w = w[:w.shape[0]//10]
    fig = sd.plot_orbits(toy_w, marker=None, linestyle='-',
                         alpha=0.5, triangle=True, c='r')
    fig = sd.plot_orbits(plot_w, axes=fig.axes, marker=None, linestyle='-',
                         alpha=0.8, triangle=True, c='k')
    fig.savefig(os.path.join(plot_path, "orbit_xyz{}.png".format(suffix)))

    # plot a smaller section of the orbit in the meridional plane
    fig,ax = plt.subplots(1,1,figsize=(6,6))
    R = np.sqrt(plot_w[:,0,0]**2 + plot_w[:,0,1]**2)
    toy_R = np.sqrt(toy_w[:,0,0]**2 + toy_w[:,0,1]**2)
    ax.plot(toy_R, toy_w[:,0,2], marker=None, linestyle='-', alpha=0.5, c='r')
    ax.plot(R, plot_w[:,0,2], marker=None, linestyle='-', alpha=0.8, c='k')
    ax.set_xlabel("$R$")
    ax.set_ylabel("$Z$", rotation='horizontal')
    fig.savefig(os.path.join(plot_path, "orbit_Rz{}.png".format(suffix)))

    return t,w,toy_potential

def make_action_files(t, w, potential, suffix="", overwrite=False,
                      force_harmonic_oscillator=False, N_max=6):

    action_filename = os.path.join(plot_path, "actions{}.npy".format(suffix))

    if overwrite and os.path.exists(action_filename):
        os.remove(action_filename)

    if not os.path.exists(action_filename):
        # compute the actions and angles for the orbit
        actions,angles,freqs = sd.cross_validate_actions(t, w[:,0], N_max=N_max, nbins=100,
                                    force_harmonic_oscillator=force_harmonic_oscillator,
                                    usys=potential.usys, skip_failures=True,
                                    overlap=0) #w.shape[0]//100)

        # now compute for the full time series
        r = sd.find_actions(t, w[:,0], N_max=N_max, usys=potential.usys, return_Sn=True,
                            force_harmonic_oscillator=force_harmonic_oscillator)
        full_actions,full_angles,full_freqs = r[:3]
        Sn,dSn_dJ,nvecs = r[3:]

        np.save(action_filename, (actions,angles,freqs) + r)
        logger.debug("Actions computed and saved to file: {}".format(action_filename))
    else:
        r = np.load(action_filename)
        actions,angles,freqs = r[:3]
        full_actions,full_angles,full_freqs = r[3:6]
        Sn,dSn_dJ,nvecs = r[6:]
        logger.debug("Actions read from file: {}".format(action_filename))

    return actions,angles,freqs,full_actions,full_angles,full_freqs

def action_plots(actions,angles,freqs,full_actions,full_angles,full_freqs,
                 t, w, toy_potential, suffix=""):

    # --------------------------------------------------------
    # deviation of actions from actions computed on full orbit
    fig,axes = plt.subplots(1,3,figsize=(12,5),sharey=True,sharex=True)
    dev_percent = (actions - full_actions[None]) / full_actions[None]*100.
    max_dev = np.max(np.abs(dev_percent))
    max_dev = float("{:.0e}".format(max_dev))
    for i in range(3):
        axes[i].set_title("$J_{}$".format(i+1), y=1.02)
        axes[i].plot(dev_percent[:,i], marker='.', linestyle='none')

    if max_dev < 0.1:
        max_dev = 0.1

    axes[0].set_ylim(-max_dev,max_dev)
    axes[0].set_yticks(np.linspace(-max_dev,max_dev,5))
    axes[0].set_yticklabels(["{}%".format(tck) for tck in axes[0].get_yticks()])
    axes[1].set_xlabel("subsample index")

    fig.suptitle("Percent deviation of subsample action value", fontsize=20)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_path, "action_hist{}.png".format(suffix)))

    # --------------------------------------------------------
    # deviation of frequencies from freqs computed on full orbit
    fig,axes = plt.subplots(1,3,figsize=(12,5),sharey=True,sharex=True)
    dev_percent = (freqs - full_freqs[None]) / full_freqs[None]*100.
    max_dev = np.max(np.abs(dev_percent))
    max_dev = float("{:.0e}".format(max_dev))
    for i in range(3):
        axes[i].set_title(r"$\Omega_{}$".format(i+1), y=1.02)
        axes[i].plot(dev_percent[:,i], marker='.', linestyle='none')

    if max_dev < 0.1:
        max_dev = 0.1

    axes[0].set_ylim(-max_dev,max_dev)
    axes[0].set_yticks(np.linspace(-max_dev,max_dev,5))
    axes[0].set_yticklabels(["{}%".format(tck) for tck in axes[0].get_yticks()])
    axes[1].set_xlabel("subsample index")

    fig.suptitle("Percent deviation of subsample frequency value", fontsize=20)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_path, "freq_hist{}.png".format(suffix)))

    # --------------------------------------------------------
    # now going to plot toy actions and solved actions
    actions,angles = toy_potential.action_angle(w[:,0,:3],w[:,0,3:])

    fig,axes = plt.subplots(1,3,figsize=(12,5),sharey=True,sharex=True)
    dev_percent = (actions-full_actions[None])/full_actions[None]*100
    max_dev = np.max(np.abs(dev_percent))
    max_dev = float("{:.0e}".format(max_dev))
    for i in range(3):
        axes[i].plot(t/1000., dev_percent[:,i], linestyle='none',
                     marker='.', alpha=0.75, label='toy action')
        axes[i].axhline(0., lw=1., zorder=-1, c='#31a354')
        axes[i].set_title("$J_{}$".format(i+1), y=1.02)

    axes[0].set_ylim(-max_dev,max_dev)
    axes[0].set_yticks(np.linspace(-max_dev,max_dev,5))
    axes[0].set_yticklabels(["{}%".format(tck) for tck in axes[0].get_yticks()])
    axes[1].set_xlabel("time [Gyr]")

    fig.suptitle("Percent deviation from estimated action", fontsize=20)
    axes[0].legend(fontsize=16)

    dt = t[1]-t[0]
    axes[0].set_xlim(0.,5.)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_path,"toy_computed_actions{}.png".format(suffix)))

def main(orbit_name, overwrite=False, N_max=6):

    # define an axisymmetric potential
    usys = (u.kpc, u.Msun, u.Myr)

    if orbit_name == "axisymmetricloop":
        suffix = "_" + orbit_name

        # well-fit loop orbit
        p = sp.LogarithmicPotential(v_c=0.15, r_h=0., phi=0.,
                                    q1=1., q2=1., q3=0.85,  usys=usys)
        w0 = [8.,0.,0.,0.075,0.15,0.05]

    elif orbit_name == "triaxialloop":
        suffix = "_" + orbit_name

        # well-fit loop orbit
        p = sp.LogarithmicPotential(v_c=0.15, r_h=0., phi=0.,
                                    q1=1.3, q2=1., q3=0.85,  usys=usys)
        w0 = [8.,0.,0.,0.05,0.175,0.05]

    elif orbit_name == "triaxialchaotic":
        suffix = "_" + orbit_name

        # chaotic orbit?
        p = sp.LogarithmicPotential(v_c=0.15, r_h=0., phi=0.,
                                    q1=1.3, q2=1., q3=0.85,  usys=usys)
        w0 = [5.5,5.5,0.,-0.02,0.02,0.11]

    else:
        raise NameError("No orbit name '{}'".format(orbit_name))

    t,w,toy_potential = make_orbit_files(p, w0, suffix=suffix,
                                         overwrite=overwrite, N_max=N_max)
    r = make_action_files(t, w, p, suffix=suffix, overwrite=overwrite, N_max=N_max)
    action_plots(*r, t=t, w=w, toy_potential=toy_potential, suffix=suffix)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")

    parser.add_argument("-o", dest="overwrite", action="store_true", default=False,
                        help="Overwrite generated files.")
    parser.add_argument("-n", "--name", dest="name", required=True,
                        help="Name of the orbit. 'loop' or 'chaotic'.")
    parser.add_argument("--Nmax", dest="Nmax", default=6, type=int,
                        help="Max. length of integer vector.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(args.name, overwrite=args.overwrite, N_max=args.Nmax)