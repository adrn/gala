# coding: utf-8

""" Test helpers """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import matplotlib.pyplot as plt
import numpy as np

def plot(ts, ws, marker='.', alpha=0.75, linestyle='-', fig=None):
    """ Make some helpful plots for testing the integrators. """

    if ws.ndim == 2:
        ws = ws[:,np.newaxis]

    nsteps,nparticles,ndim = ws.shape
    if fig is None:
        fig,axes = plt.subplots(ndim, ndim, figsize=(4*ndim,4*ndim))
    else:
        axes = np.array(fig.axes).reshape(ndim,ndim)

    kwargs = dict(marker=marker, linestyle=linestyle, alpha=alpha)
    for ii in range(ndim):
        for jj in range(ndim):
            if ii == jj:
                axes[jj,ii].plot(ws[:,0,ii], **kwargs)
            else:
                axes[jj,ii].plot(ws[:,0,ii], ws[:,0,jj], **kwargs)

    #fig.tight_layout()

    for ii in range(ndim):
        for jj in range(ndim):
            if ii > jj:
                axes[jj,ii].set_visible(False)
                continue
    return fig