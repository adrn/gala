# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: gala
#     language: python
#     name: python3
# ---

# %% nbsphinx="hidden"
# %matplotlib inline

# %% nbsphinx="hidden"
# %run ../tutorials/nb_setup

# %% [markdown]
# # Defining the MilkyWayPotential model
#
# ## Introduction
#
# `gala` provides simplified mass models for the Milky Way to use in orbit integration or dynamical calculations. Some of these mass models come from other publications or packages (e.g., the Law and Majewski 2010 model `LM10Potential`). Some of the potential models are defined and provided by Gala. This document describes how we determined the parameters of the Gala Milky Way models.
#
# We determine parameters of the Gala Milky Way models using compilations of enclosed mass measurements of the Milky Way and measurements of the mass structure of the Galactic disk. We then fit for the parameters of a multi-component model (e.g., disk, bulge, halo, etc.) using these measurements.

# %%
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import G
from astropy.io import ascii
from scipy.optimize import leastsq

import gala.potential as gp
from gala.units import galactic

# %% [markdown]
# ## `MilkyWayPotential` version 1 (circa 2017)
#
# This model was previously just known as `MilkyWayPotential` in Gala, now known as "version 1," and represents an older model based on measurements that are now out of date. We still describe the process of fitting for this model, for completeness.
#
# The source data for this model was compiled from published values and is included with Gala:

# %%
mwdata1 = ascii.read("data/MW_mass_enclosed.csv")
mwdata1

# %% [markdown]
# We can now plot the above data and uncertainties:

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout="tight")

ax.errorbar(
    mwdata1["r"],
    mwdata1["Menc"],
    yerr=(mwdata1["Menc_err_neg"], mwdata1["Menc_err_pos"]),
    marker="o",
    markersize=2,
    color="k",
    alpha=1.0,
    ecolor="#aaaaaa",
    capthick=0,
    linestyle="none",
    elinewidth=1.0,
)

ax.set_xlim(1e-3, 10**2.6)
ax.set_ylim(7e6, 10**12.25)

ax.set_xlabel("$r$ [kpc]")
ax.set_ylabel(r"$M(<r)$ [M$_\odot$]")

ax.set_xscale("log")
ax.set_yscale("log")

# %% [markdown]
# We now need to assume some form for the potential. We use a four component potential model consisting of a Hernquist
# ([1990](https://ui.adsabs.harvard.edu/#abs/1990ApJ...356..359H/abstract))
# bulge and nucleus, a Miyamoto-Nagai
# ([1975](https://ui.adsabs.harvard.edu/#abs/1975PASJ...27..533M/abstract))
# disk, and an NFW
# ([1997](https://ui.adsabs.harvard.edu/#abs/1997ApJ...490..493N/abstract))
# halo. We fix the parameters of the disk and bulge to be consistent with
# previous work ([Bovy
# 2015](https://ui.adsabs.harvard.edu/#abs/2015ApJS..216...29B/abstract) -
# please cite that paper if you use this potential model) and vary the scale
# mass and scale radius of the nucleus and halo, respectively. We'll fit for
# these parameters in log-space, so we'll first define a function that returns a
# `gala.potential.CCompositePotential` object given these four parameters:


# %%
def get_potential(log_M_h, log_r_s, log_M_n, log_a):
    mw_potential = gp.CCompositePotential()
    mw_potential["bulge"] = gp.HernquistPotential(m=5e9, c=1.0, units=galactic)
    mw_potential["disk"] = gp.MiyamotoNagaiPotential(
        m=6.8e10 * u.Msun, a=3 * u.kpc, b=280 * u.pc, units=galactic
    )
    mw_potential["nucl"] = gp.HernquistPotential(
        m=np.exp(log_M_n), c=np.exp(log_a) * u.pc, units=galactic
    )
    mw_potential["halo"] = gp.NFWPotential(
        m=np.exp(log_M_h), r_s=np.exp(log_r_s), units=galactic
    )

    return mw_potential


# %% [markdown]
# We now need to specify an initial guess for the parameters - let's do that (by
# making them up), and then plot the initial guess potential over the data:

# %%
# Initial guess for the parameters- units are:
#     [Msun, kpc, Msun, pc]
x0 = [np.log(6e11), np.log(20.0), np.log(2e9), np.log(100.0)]
init_potential = get_potential(*x0)

# %%
r = np.logspace(-3, 3, 256) * u.kpc

fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout="tight")

ax.errorbar(
    mwdata1["r"],
    mwdata1["Menc"],
    yerr=(mwdata1["Menc_err_neg"], mwdata1["Menc_err_pos"]),
    marker="o",
    markersize=2,
    color="k",
    alpha=1.0,
    ecolor="#aaaaaa",
    capthick=0,
    linestyle="none",
    elinewidth=1.0,
)

# Use symmetry coordinates for spherical mass_enclosed calculation
fit_menc = init_potential.mass_enclosed(R=r)
ax.loglog(r.value, fit_menc.value, marker="", color="#3182bd", linewidth=2, alpha=0.7)

ax.set_xlim(1e-3, 10**2.6)
ax.set_ylim(7e6, 10**12.25)

ax.set_xlabel("$r$ [kpc]")
ax.set_ylabel(r"$M(<r)$ [M$_\odot$]")

ax.set_xscale("log")
ax.set_yscale("log")

# %% [markdown]
# It looks pretty good already! But let's now use least-squares fitting to
# optimize our nucleus and halo parameters. We first need to define an error
# function:


# %%
def err_func(p, r, Menc, Menc_err):
    pot = get_potential(*p)
    model_menc = pot.mass_enclosed(R=r * u.kpc).to(u.Msun).value
    return (model_menc - Menc) / Menc_err


# %% [markdown]
# Because the uncertainties are all approximately but not exactly symmetric,
# we'll take the maximum of the upper and lower uncertainty values and assume
# that the uncertainties in the mass measurements are Gaussian (a bad but simple
# assumption):

# %%
err = np.max([mwdata1["Menc_err_pos"], mwdata1["Menc_err_neg"]], axis=0)
p_opt, ier = leastsq(err_func, x0=x0, args=(mwdata1["r"], mwdata1["Menc"], err))
assert ier in range(1, 4 + 1), "least-squares fit failed!"
fit_potential = get_potential(*p_opt)

# %% [markdown]
# Now we have a best-fit potential! Let's plot the enclosed mass of the fit potential over the data:

# %%
r = np.logspace(-3, 3, 256) * u.kpc

fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout="tight")

ax.errorbar(
    mwdata1["r"],
    mwdata1["Menc"],
    yerr=(mwdata1["Menc_err_neg"], mwdata1["Menc_err_pos"]),
    marker="o",
    markersize=2,
    color="k",
    alpha=1.0,
    ecolor="#aaaaaa",
    capthick=0,
    linestyle="none",
    elinewidth=1.0,
)

fit_menc = fit_potential.mass_enclosed(R=r)
ax.loglog(r.value, fit_menc.value, marker="", color="#3182bd", linewidth=2, alpha=0.7)

ax.set_xlim(1e-3, 10**2.6)
ax.set_ylim(7e6, 10**12.25)

ax.set_xlabel("$r$ [kpc]")
ax.set_ylabel(r"$M(<r)$ [M$_\odot$]")

ax.set_xscale("log")
ax.set_yscale("log")

# %% [markdown]
# This potential model can be loaded and used in Gala with:
# ```
# import gala.potential as gp
# potential = gp.MilkyWayPotential(version="v1")
# ```

# %% [markdown]
# ---
#
# ## `MilkyWayPotential` version 2 (circa 2022)
#
# This model was previously known as `MilkyWayPotential2022` in Gala, now known as `MilkyWayPotential` "version 2," and represents an updated model of the Milky Way based on more recent measurements of the halo and disk structure.
#
# The source data for this model was compiled from published values and is included with Gala — here we use the Eilers et al. (2019) circular velocity measurements from APOGEE for mass enclosed measurements within the disk region:

# %%
eilers = ascii.read("data/Eilers2019-circ-velocity.txt")

mwdata2 = mwdata1.copy()

# remove old measurements within disk region:
mwdata2.remove_rows([2, 3, 4])

# %% [markdown]
# Here we convert the circular velocity measurements to enclosed mass measurements, to be consistent with the other input data. We also adopt a 5% uncertainty on the enclosed mass measurements:

# %%
eilers_r = eilers["R"][eilers["R"] < 15] * u.kpc
eilers_v = eilers["v_c"][eilers["R"] < 15] * u.km / u.s
eilers_M = (eilers_v**2 * eilers_r / G).to(u.Msun)
for r, M in zip(eilers_r, eilers_M):
    mwdata2.add_row(
        [r.value, M.value, 0.05 * M.value, 0.05 * M.value, "Eilers et al. (2019)"]
    )

# %%
r = np.logspace(-3, 3, 256) * u.kpc

fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout="tight")

ax.errorbar(
    mwdata2["r"],
    mwdata2["Menc"],
    yerr=(mwdata2["Menc_err_neg"], mwdata2["Menc_err_pos"]),
    marker="o",
    markersize=2,
    color="k",
    alpha=1.0,
    ecolor="#aaaaaa",
    capthick=0,
    linestyle="none",
    elinewidth=1.0,
)

ax.set_xlim(1e-3, 10**2.6)
ax.set_ylim(7e6, 10**12.25)

ax.set_xlabel("$r$ [kpc]")
ax.set_ylabel(r"$M(<r)$ [M$_\odot$]")

ax.set_xscale("log")
ax.set_yscale("log")


# %% [markdown]
# For hte new model, we use the same bulge component as with v1, but we now use a mixture of Miyamoto-Nagai disk models to represent an approximately radially exponential and vertically sech^2 disk density profile:


# %%
def get_potential_v2(log_M_h, log_r_s, log_M_n, log_a, log_M_d):
    mw_potential = gp.CCompositePotential()

    # Fixed bulge component
    mw_potential["bulge"] = gp.HernquistPotential(m=5e9, c=1.0, units=galactic)

    # Scale radius and height are fixed to Bland-Hawthorn & Gerhard (2016) values
    mw_potential["disk"] = gp.MN3ExponentialDiskPotential(
        m=np.exp(log_M_d) * u.Msun,
        h_R=2.6 * u.kpc,
        h_z=300 * u.pc,
        units=galactic,
        sech2_z=False,
    )

    mw_potential["nucl"] = gp.HernquistPotential(
        m=np.exp(log_M_n), c=np.exp(log_a) * u.pc, units=galactic
    )
    mw_potential["halo"] = gp.NFWPotential(
        m=np.exp(log_M_h), r_s=np.exp(log_r_s), units=galactic
    )

    return mw_potential


# %%
x0 = [np.log(6e11), np.log(20.0), np.log(2e9), np.log(100.0), np.log(4e10)]
init_potential_v2 = get_potential_v2(*x0)


# %%
def err_func_v2(p, r, Menc, Menc_err):
    pot = get_potential_v2(*p)
    xyz = np.zeros((3, len(r)))
    xyz[0] = r
    model_menc = pot.mass_enclosed(xyz).to(u.Msun).value
    return (model_menc - Menc) / Menc_err


# %%
err = np.max([mwdata2["Menc_err_pos"], mwdata2["Menc_err_neg"]], axis=0)
p_opt, ier = leastsq(err_func_v2, x0=x0, args=(mwdata2["r"], mwdata2["Menc"], err))
assert ier in range(1, 4 + 1), "least-squares fit failed!"

fit_potential_v2 = get_potential_v2(*p_opt)

# %% [markdown]
# Here is the fitted potential compared to the data:

# %%
r = np.logspace(-3, 3, 256) * u.kpc

fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout="tight")

ax.errorbar(
    mwdata2["r"],
    mwdata2["Menc"],
    yerr=(mwdata2["Menc_err_neg"], mwdata2["Menc_err_pos"]),
    marker="o",
    markersize=2,
    color="k",
    alpha=1.0,
    ecolor="#aaaaaa",
    capthick=0,
    linestyle="none",
    elinewidth=1.0,
)

fit_menc = fit_potential_v2.mass_enclosed(R=r)
ax.loglog(r.value, fit_menc.value, marker="", color="#3182bd", linewidth=2, alpha=0.7)

ax.set_xlim(1e-3, 10**2.6)
ax.set_ylim(7e6, 10**12.25)

ax.set_xlabel("$r$ [kpc]")
ax.set_ylabel(r"$M(<r)$ [M$_\odot$]")

ax.set_xscale("log")
ax.set_yscale("log")

# %% [markdown]
# And a comparison of the circular velocity curve between v1 and v2, compared to the Eilers et al. (2019) data:

# %%
mwpot_v1 = gp.MilkyWayPotential(version="v1")

r_grid = np.linspace(0.5, 30, 128)

fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout="tight")
ax.plot(
    r_grid,
    fit_potential_v2.circular_velocity(R=r_grid),
    marker="",
    lw=2,
    label="v2",
    color="tab:blue",
)
ax.plot(
    r_grid,
    mwpot_v1.circular_velocity(R=r_grid),
    marker="",
    label="v1",
    linestyle="--",
    color="tab:orange",
)
ax.scatter(
    eilers["R"][eilers["R"] < 15],
    eilers["v_c"][eilers["R"] < 15],
    marker="o",
    s=8,
    color="k",
)
ax.legend(loc="best", fontsize=16)
ax.set_xlabel("$R$ [kpc]")
ax.set_ylabel("$v_c$")

# %% [markdown]
# Note that the parameters of this fit were further tweaked (as described in [Hunt et al. 2022](https://ui.adsabs.harvard.edu/abs/2022MNRAS.516L...7H/abstract)) to provide a better match to the vertical phase spiral morphology.
