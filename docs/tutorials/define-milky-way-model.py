# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + nbsphinx="hidden"
# %matplotlib inline

# + nbsphinx="hidden"
# %run nb_setup
# -

# # Defining a Milky Way potential model

# +
# Third-party dependencies
from astropy.io import ascii
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

# Gala
import gala.potential as gp
from gala.units import galactic
# -

# ## Introduction
#
# `gala` provides a simple and easy way to access and integrate orbits in an
# approximate mass model for the Milky Way. The parameters of the mass model are
# determined by least-squares fitting the enclosed mass profile of a pre-defined
# potential form to recent measurements compiled from the literature. These
# measurements are provided with the documentation of `gala` and are shown
# below. The radius units are kpc, and mass units are solar masses:

tbl = ascii.read('data/MW_mass_enclosed.csv')

tbl

# Let's now plot the above data and uncertainties:

# +
fig, ax = plt.subplots(1, 1, figsize=(4, 4))

ax.errorbar(tbl['r'], tbl['Menc'], yerr=(tbl['Menc_err_neg'],
                                         tbl['Menc_err_pos']),
            marker='o', markersize=2, color='k', alpha=1., ecolor='#aaaaaa',
            capthick=0, linestyle='none', elinewidth=1.)

ax.set_xlim(1E-3, 10**2.6)
ax.set_ylim(7E6, 10**12.25)

ax.set_xlabel('$r$ [kpc]')
ax.set_ylabel('$M(<r)$ [M$_\odot$]')

ax.set_xscale('log')
ax.set_yscale('log')

fig.tight_layout()


# -

# We now need to assume some form for the potential. For simplicity and within
# reason, we'll use a four component potential model consisting of a Hernquist
# ([1990](https://ui.adsabs.harvard.edu/#abs/1990ApJ...356..359H/abstract))
# bulge and nucleus, a Miyamoto-Nagai
# ([1975](https://ui.adsabs.harvard.edu/#abs/1975PASJ...27..533M/abstract))
# disk, and an NFW
# ([1997](https://ui.adsabs.harvard.edu/#abs/1997ApJ...490..493N/abstract))
# halo. We'll fix the parameters of the disk and bulge to be consistent with
# previous work ([Bovy
# 2015](https://ui.adsabs.harvard.edu/#abs/2015ApJS..216...29B/abstract) -
# please cite that paper if you use this potential model) and vary the scale
# mass and scale radius of the nucleus and halo, respectively. We'll fit for
# these parameters in log-space, so we'll first define a function that returns a
# `gala.potential.CCompositePotential` object given these four parameters:

def get_potential(log_M_h, log_r_s, log_M_n, log_a):
    mw_potential = gp.CCompositePotential()
    mw_potential['bulge'] = gp.HernquistPotential(m=5E9, c=1., units=galactic)
    mw_potential['disk'] = gp.MiyamotoNagaiPotential(m=6.8E10*u.Msun, a=3*u.kpc, b=280*u.pc,
                                                     units=galactic)
    mw_potential['nucl'] = gp.HernquistPotential(m=np.exp(log_M_n), c=np.exp(log_a)*u.pc,
                                                 units=galactic)
    mw_potential['halo'] = gp.NFWPotential(m=np.exp(log_M_h), r_s=np.exp(log_r_s), units=galactic)

    return mw_potential


# We now need to specify an initial guess for the parameters - let's do that (by
# making them up), and then plot the initial guess potential over the data:

# Initial guess for the parameters- units are:
#     [Msun, kpc, Msun, pc]
x0 = [np.log(6E11), np.log(20.), np.log(2E9), np.log(100.)]
init_potential = get_potential(*x0)

# +
xyz = np.zeros((3, 256))
xyz[0] = np.logspace(-3, 3, 256)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))

ax.errorbar(tbl['r'], tbl['Menc'], yerr=(tbl['Menc_err_neg'], tbl['Menc_err_pos']),
            marker='o', markersize=2, color='k', alpha=1., ecolor='#aaaaaa',
            capthick=0, linestyle='none', elinewidth=1.)

fit_menc = init_potential.mass_enclosed(xyz*u.kpc)
ax.loglog(xyz[0], fit_menc.value, marker='', color="#3182bd",
          linewidth=2, alpha=0.7)

ax.set_xlim(1E-3, 10**2.6)
ax.set_ylim(7E6, 10**12.25)

ax.set_xlabel('$r$ [kpc]')
ax.set_ylabel('$M(<r)$ [M$_\odot$]')

ax.set_xscale('log')
ax.set_yscale('log')

fig.tight_layout()


# -

# It looks pretty good already! But let's now use least-squares fitting to
# optimize our nucleus and halo parameters. We first need to define an error
# function:

def err_func(p, r, Menc, Menc_err):
    pot = get_potential(*p)
    xyz = np.zeros((3, len(r)))
    xyz[0] = r
    model_menc = pot.mass_enclosed(xyz).to(u.Msun).value
    return (model_menc - Menc) / Menc_err


# Because the uncertainties are all approximately but not exactly symmetric,
# we'll take the maximum of the upper and lower uncertainty values and assume
# that the uncertainties in the mass measurements are Gaussian (a bad but simple
# assumption):

err = np.max([tbl['Menc_err_pos'], tbl['Menc_err_neg']], axis=0)
p_opt, ier = leastsq(err_func, x0=x0, args=(tbl['r'], tbl['Menc'], err))
assert ier in range(1, 4+1), "least-squares fit failed!"
fit_potential = get_potential(*p_opt)

# Now we have a best-fit potential! Let's plot the enclosed mass of the fit potential over the data:

# +
xyz = np.zeros((3, 256))
xyz[0] = np.logspace(-3, 3, 256)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))

ax.errorbar(tbl['r'], tbl['Menc'], yerr=(tbl['Menc_err_neg'], tbl['Menc_err_pos']),
            marker='o', markersize=2, color='k', alpha=1., ecolor='#aaaaaa',
            capthick=0, linestyle='none', elinewidth=1.)

fit_menc = fit_potential.mass_enclosed(xyz*u.kpc)
ax.loglog(xyz[0], fit_menc.value, marker='', color="#3182bd",
          linewidth=2, alpha=0.7)

ax.set_xlim(1E-3, 10**2.6)
ax.set_ylim(7E6, 10**12.25)

ax.set_xlabel('$r$ [kpc]')
ax.set_ylabel('$M(<r)$ [M$_\odot$]')

ax.set_xscale('log')
ax.set_yscale('log')

fig.tight_layout()
# -

# This potential is already implemented in `gala` in `gala.potential.special`,
# and we can import it with:

from gala.potential import MilkyWayPotential

potential = MilkyWayPotential()
potential
