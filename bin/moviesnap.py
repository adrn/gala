#!/usr/bin/env python
# coding: utf-8

""" Make a movie from a bunch of SNAP files. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
from astropy import log as logger
import astropy.units as u
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Project
import streamteam.dynamics as sd
import streamteam.io as sio
from streamteam.units import galactic

def anim_update(num, scf, snapfiles, lines, title):
    snapfile = os.path.split(snapfiles[num])[1]
    tbl = scf.read_snap(snapfile, units=usys)
    w = tbl_to_w(tbl)

    pct_bound = sum(tbl['tub'] == 0) / float(len(w))
    ttext = "Time: {:05.2f} Gyr, Percent bound: {:04.1f}"\
            .format(tbl.meta['time']/1000., pct_bound)
    title.set_text()

    data = w[:,0,:2].T.copy()
    lines[0].set_data(data)
    data = w[:,0,np.array([0,2])].T.copy()
    lines[1].set_data(data)
    data = w[:,0,np.array([1,2])].T.copy()
    lines[2].set_data(data)
    return lines

def main(snap_path, plot_cen=False, bound=100., output="orbit.mp4"):
    path,snap_file = os.path.split(snap_path)

    scf = sio.SCFReader(path)

    # get list of all SNAP files, excluding png's if there are any
    snapfiles = [f for f in glob.glob(os.path.join(path, "SNAP*"))
                   if os.path.splitext(f)[1] != ".png"]

    style = dict(alpha=0.05, marker='.', linestyle='none', color='k')
    fig,axes = plt.subplots(1, 3, figsize=(14,5), sharex=True, sharey=True)
    l0, = axes[0].plot([], [], **style)
    l1, = axes[1].plot([], [], **style)
    l2, = axes[2].plot([], [], **style)
    lines = [l0,l1,l2]

    if plot_cen:
        cen_w = sio.tbl_to_w(scf.read_cen(galactic))
        faxes[0].plot(cen_w[...,0], cen_w[...,1], marker=None,
                     linestyle='-', zorder=-100, color='#2b8cbe')
        axes[1].plot(cen_w[...,0], cen_w[...,2], marker=None,
                     linestyle='-', zorder=-100, color='#2b8cbe')
        axes[2].plot(cen_w[...,1], cen_w[...,2], marker=None,
                     linestyle='-', zorder=-100, color='#2b8cbe')

    axes[0].set_xlabel("X [kpc]")
    axes[0].set_ylabel("Y [kpc]")
    axes[1].set_xlabel("X [kpc]")
    axes[1].set_ylabel("Z [kpc]")
    axes[2].set_xlabel("Y [kpc]")
    axes[2].set_ylabel("Z [kpc]")

    axes[0].set_xlim(-bound,bound)
    axes[0].set_ylim(-bound,bound)

    title = fig.suptitle("")
    anim = FuncAnimation(fig, anim_update, len(snapfiles),
                         fargs=(scf, snapfiles, lines, title),
                         interval=50, blit=True)

    anim.save(os.path.join(path, output), bitrate=2048)

if __name__ == '__main__':
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")

    parser.add_argument("-p", "--path", dest="snap_path", required=True,
                        type=str, help="Path to SNAP file.")
    parser.add_argument("--cen", dest="plot_cen", action="store_true",
                        default=False, help="Plot SCFCEN orbit.")
    parser.add_argument("--bound", dest="bound", type=float,
                        help="Bounding distance.")
    parser.add_argument("-o", "--output", dest="output", default="orbit.mp4",
                        type=str, help="Output file.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(args.snap_path, args.plot_cen, bound=args.bound, output=args.output)