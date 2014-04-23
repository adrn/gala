# coding: utf-8

""" Test helpers """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import matplotlib.pyplot as plt
import numpy as np

def plot(ts, q, p):
    """ Make some helpful plots for testing the integrators. """

    qp = np.squeeze(np.vstack((q.T,p.T)))

    ndim = qp.shape[0]
    fig,axes = plt.subplots(ndim, ndim, figsize=(4*ndim,4*ndim))

    kwargs = dict(marker='.', linestyle='-')
    for ii in range(ndim):
        for jj in range(ndim):

            if ii == jj:
                axes[jj,ii].plot(qp[ii], linestyle='-',
                                 marker='.', alpha=0.75)
            else:
                axes[jj,ii].plot(qp[ii], qp[jj], linestyle='-',
                                 marker='.', alpha=0.75)

    fig.tight_layout()

    for ii in range(ndim):
        for jj in range(ndim):
            if ii > jj:
                axes[jj,ii].set_visible(False)
                continue
    return fig