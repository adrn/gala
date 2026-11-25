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
#     display_name: gala (3.12.10)
#     language: python
#     name: python3
# ---

# %% nbsphinx="hidden"
# %run nb_setup

# %%
# %matplotlib inline

# %% [markdown]
# # Stochastic diffusion in orbit integrations
#
# Stars in the galactic disk are subject to stochastic forces from fluctuations of the gravitational field. This can be from giant molecular clouds (or really ISM structure), passing stars, dark-matter substructure, etc. If the forces are random and averaged over many encounters, these perturbations act as a *diffusion* of the stars' orbits. Gala can include this effect in an orbit integration by applying small random kicks to orbits.
#
# In this tutorial we:
#
# 1. describe how the stochastic kick works and how to set it up,
# 2. demonstrate it on a particle in free-space, and
# 3. use it to show how vertical diffusion blurs a "phase spiral" (the "Gaia snail")
#    in the vertical phase space.
#
# ### Notebook Setup and Package Imports

# %%
from itertools import pairwise

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

import gala.dynamics as gd
import gala.potential as gp
from gala.dynamics.diffusion import ConstantDiffusion
from gala.units import galactic

# %% [markdown]
# ## Modeling stochastic diffusion
#
# A diffusion model specifies, at each phase-space point $\boldsymbol{w}$, a drift
# vector $\boldsymbol{\mu}$ and a $6\times6$ diffusion tensor $\mathbf{D}$. After every
# leapfrog step of length $\delta t$, each orbit receives an Euler-Maruyama kick
#
# $$
# \delta\boldsymbol{w} = \boldsymbol{\mu}\,\delta t + \mathbf{L}\,\sqrt{\delta t}\,
# \boldsymbol{\xi}, \qquad \mathbf{L}\mathbf{L}^\mathsf{T} = \mathbf{D},
# $$
#
# where $\boldsymbol{\xi}$ is a vector of independent standard normal draws. So the
# covariance of the kicks grows as $\mathbf{D}\,t$. A few implementation details:
#
# - Diffusion is supported by the C leapfrog integrator only.
# - The random number draws are generated from a seed, specified with the `kick_seed`
#   argument, so if you pass an explicit seed it will make the diffusion reproducible.
# - The diffusion coefficients are specified as raw numbers in the potential's unit
#   system. This is because the elements of the diffusion tensor have different units for
#   the position and velocity terms. For example, with `galactic` units, a
#   velocity-diffusion entry is in $(\mathrm{kpc/Myr})^2/\mathrm{Myr}$.
# - The coefficients can be given in Cartesian or cylindrical coordinates $(R, \phi, z,
#   v_R, v_\phi, v_z)$, where the $\phi$ entry is actually the physical arc length
#   $R\,\delta\phi$. You probably want to use cylindrical coordinates for a disk galaxy because the tensor is more likely to be nearly constant.
# - The kicks happen at regulary intervals. By default, every timestep - but this is very
#   inefficient. You can safely set `kick_every=k` to apply the kick every $k$ steps,
#   typically with $k \approx 10$--100.
#
# Here we use `ConstantDiffusion`, which has the same $\boldsymbol\mu$ and $\mathbf{D}$
# everywhere. You can also use the `GriddedDiffusion` class to interpolate the values over a grid in $(R, |z|)$.

# %% [markdown]
# ## A particle diffusing in free space
#
# Here we start many particles at rest with no potential but finite diffusion coefficient in $v_z$. This is a simple test of the diffusion implementation: The variance of $v_z$ should grow as $D_{v_zv_z}\,t$:

# %%
N = 5000
D = np.zeros((6, 6))
D[5, 5] = 1e-6  # (kpc/Myr)^2 / Myr
free_diffusion = ConstantDiffusion(D=D, basis="cartesian", units=galactic)

H_free = gp.Hamiltonian(gp.NullPotential(units=galactic))
w0_free = gd.PhaseSpacePosition(
    pos=np.zeros((3, N)) * u.kpc, vel=np.zeros((3, N)) * u.kpc / u.Myr
)

times, variances = [], []
w, t = w0_free, 0.0
for i in range(10):
    orbit = H_free.integrate_orbit(
        w,
        dt=1.0,
        t1=t,
        n_steps=20,
        save_all=False,
        diffusion=free_diffusion,
        kick_seed=(42, i),
    )
    w, t = orbit[-1], t + 20
    times.append(t)
    variances.append(np.var(w.v_z.to_value(u.kpc / u.Myr)))

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(times, variances, marker="o", ls="none", label="simulation")
ax.plot(times, D[5, 5] * np.array(times), color="k", label=r"$D_{v_zv_z}\,t$")
ax.set(xlabel="$t$ [Myr]", ylabel=r"Var$(v_z)$ [(kpc/Myr)$^2$]")
ax.legend()

# %% [markdown]
# ## A phase spiral with diffusion
#
# A phase spiral forms when a population of stars is kicked simultaneously. The resulting distribution function then phase mixes to form a spiral. If the kick is predominantly vertical, the stars form a vertical phase spiral, like the one seen in Gaia data. Here we set up a population of stars on nearly circular orbits with radii between 7.8 and 8.2 kpc in the `MilkyWayPotential`. We draw their vertical initial conditions assuming the epicyclic (harmonic) approximation, with vertical actions $J_z$ exponentially distributed (so the population is roughly isothermal with $\sigma_z \approx 15~\mathrm{km/s}$) and uniform vertical phases.

# %%
pot = gp.MilkyWayPotential(version="v2", units=galactic)

N = 40_000
rng = np.random.default_rng(8)
R0 = rng.uniform(7.8, 8.2, N)
phi0 = rng.uniform(0, 2 * np.pi, N)
vc = pot.circular_velocity(R=R0 * u.kpc).to_value(u.kpc / u.Myr)

nu = np.sqrt(pot.hessian([8.0, 0, 0] * u.kpc)[2, 2, 0]).to_value(1 / u.Myr)
sigma_z = (15 * u.km / u.s).to_value(u.kpc / u.Myr)
Jz = rng.exponential(sigma_z**2 / nu, N)
theta_z = rng.uniform(0, 2 * np.pi, N)
z0 = np.sqrt(2 * Jz / nu) * np.sin(theta_z)
vz0 = np.sqrt(2 * nu * Jz) * np.cos(theta_z)

w0_init = gd.PhaseSpacePosition(
    pos=[R0 * np.cos(phi0), R0 * np.sin(phi0), z0] * u.kpc,
    vel=[-vc * np.sin(phi0), vc * np.cos(phi0), vz0] * u.kpc / u.Myr,
)
_ = w0_init.plot(s=1.0, alpha=0.5)

# %% [markdown]
# Now we perturb the population by adding $\Delta v_z = 10~\mathrm{km/s}$ to every star's vertical velocity $v_z$:

# %%
dvz = (10 * u.km / u.s).to_value(u.kpc / u.Myr)
w0 = gd.PhaseSpacePosition(
    pos=[R0 * np.cos(phi0), R0 * np.sin(phi0), z0] * u.kpc,
    vel=[-vc * np.sin(phi0), vc * np.cos(phi0), vz0 + dvz] * u.kpc / u.Myr,
)

# %% [markdown]
# We integrate the same initial conditions three times: with no diffusion, and with
# weak, and strong diffusion in $v_z$ only. The diffusion strengths below correspond
# to a random walk in $v_z$ of about 1 and 3 km/s per 100 Myr. This is small compared to the vertical velocity dispersion, but the phase spiral structure is relatively sensitive to small perturbations.
#
# We save only the snapshots we plot, by integrating between them with
# `save_all=False`.


# %%
def vz_diffusion(kms_per_100Myr):
    sigma = (kms_per_100Myr * u.km / u.s).to_value(u.kpc / u.Myr)
    D = np.zeros((6, 6))
    D[5, 5] = sigma**2 / 100.0  # (kpc/Myr)^2 / Myr
    return ConstantDiffusion(D=D, basis="cylindrical", units=galactic)


runs = {
    "no diffusion": None,
    "weak diffusion": vz_diffusion(1.0),
    "strong diffusion": vz_diffusion(3.0),
}
snap_times = [0, 150, 300, 600]  # Myr
dt = 0.5  # Myr

snapshots = {}
for name, diffusion in runs.items():
    snapshots[name] = [w0]
    w = w0
    for i, (ta, tb) in enumerate(pairwise(snap_times)):
        orbit = pot.integrate_orbit(
            w,
            dt=dt,
            t1=ta,
            n_steps=round((tb - ta) / dt),
            save_all=False,
            diffusion=diffusion,
            kick_seed=(1, i),
        )
        w = orbit[-1]
        snapshots[name].append(w)

# %% [markdown]
# Here are the snapshots in the $(z, v_z)$ plane. Without diffusion, the spiral winds
# up tighter and tighter. With diffusion, its arms are increasingly smeared out.

# %%
fig, axes = plt.subplots(
    3,
    len(snap_times),
    figsize=(12, 9),
    sharex=True,
    sharey=True,
    layout="constrained",
)
bins = (np.linspace(-1.2, 1.2, 101), np.linspace(-50, 50, 101))
for row, name in zip(axes, runs):
    for ax, t, w in zip(row, snap_times, snapshots[name]):
        ax.hist2d(
            w.z.to_value(u.kpc),
            w.v_z.to_value(u.km / u.s),
            bins=bins,
            cmap="Greys",
            rasterized=True,
        )
        ax.set_title(f"{name}, $t = {t}$ Myr", fontsize=10)
for ax in axes[-1]:
    ax.set_xlabel("$z$ [kpc]")
for ax in axes[:, 0]:
    ax.set_ylabel("$v_z$ [km/s]")

# %% [markdown]
# As we can see, the diffusion can have a strong effect on the fine structure of the spiral, even though the heating is relatively small.
#
# This is a toy example in which the diffusion is constant and only acts in $v_z$. In reality, the diffusion is more complicated (see, e.g., Modak et al. 2026).
