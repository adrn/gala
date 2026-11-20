# ---
# jupyter:
#   jupytext:
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
# %run nb_setup

# %%
# %matplotlib inline

# %% [markdown]
# # Stochastic diffusion in orbit integrations: blurring a phase spiral
#
# Stars in galactic discs are continually nudged by fluctuations in the gravitational
# potential: giant molecular clouds, the turbulent interstellar medium, dark-matter
# substructure, and so on. Averaged over many such encounters, these perturbations act
# as a *diffusion* of the stars' orbits. Gala can include this effect in an orbit
# integration by applying a small random kick to every orbit at every step.
#
# In this tutorial we:
#
# 1. describe how the stochastic kick works and how to set it up,
# 2. check it on a free particle, and
# 3. use it to show how vertical diffusion blurs a "phase spiral" (the "Gaia snail")
#    in the $(z, v_z)$ plane of the solar neighbourhood.
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
# ## How the kick works
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
# covariance of the kicks grows as $\mathbf{D}\,t$. A few practical points:
#
# - Diffusion is supported by the C leapfrog integrator, so pass a diffusion model to
#   `integrate_orbit(..., diffusion=...)` (the default integrator for static potentials).
# - `kick_seed` makes the noise reproducible. With the default (`None`) every call draws
#   fresh noise.
# - The coefficients are **plain numbers in the potential's unit system**. With
#   `galactic` units, a velocity-diffusion entry is in $(\mathrm{kpc/Myr})^2/\mathrm{Myr}$.
#   (The tensor mixes position and velocity units, so it can't be a single `Quantity`.)
# - The coefficients can be given in Cartesian or cylindrical coordinates
#   $(R, \phi, z, v_R, v_\phi, v_z)$, where the $\phi$ entry is the physical arc length
#   $R\,\delta\phi$.
# - `kick_every=k` applies the kick every $k$ steps (with the interval since the last
#   kick as its timestep), which is cheaper when the coefficients vary slowly.
#
# Here we use `ConstantDiffusion`, which has the same $\boldsymbol\mu$ and $\mathbf{D}$
# everywhere. `GriddedDiffusion` interpolates them over a grid in $(R, |z|)$.

# %% [markdown]
# ## A free-particle check
#
# Start many particles at rest with no potential, and diffuse only $v_z$. The variance
# of $v_z$ should then grow as $D_{v_zv_z}\,t$:

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
# ## A phase spiral in the solar neighbourhood
#
# A phase spiral forms when a population of disc stars is pushed out of vertical
# equilibrium. Stars with larger vertical oscillations have lower vertical
# frequencies, so the displaced population winds up into a spiral in $(z, v_z)$. Any
# process that scatters the stars' vertical phases and amplitudes washes that spiral
# out over time.
#
# We set up stars on nearly circular orbits with radii between 7.8 and 8.2 kpc in
# the `MilkyWayPotential`. Their vertical motion is drawn in the epicyclic
# (harmonic) approximation, with vertical actions $J_z$ exponentially distributed
# (so the population is roughly isothermal with $\sigma_z \approx 15~\mathrm{km/s}$)
# and uniform vertical phases.

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

# %% [markdown]
# Now we perturb the population by adding $15~\mathrm{km/s}$ to every star's $v_z$, as
# a passing satellite might.

# %%
dvz = (15 * u.km / u.s).to_value(u.kpc / u.Myr)
w0 = gd.PhaseSpacePosition(
    pos=[R0 * np.cos(phi0), R0 * np.sin(phi0), z0] * u.kpc,
    vel=[-vc * np.sin(phi0), vc * np.cos(phi0), vz0 + dvz] * u.kpc / u.Myr,
)

# %% [markdown]
# We integrate the same initial conditions three times: with no diffusion, and with
# weak and strong diffusion in $v_z$ only. The diffusion strengths below correspond
# to a random walk in $v_z$ of about 1 and 3 km/s per 100 Myr. That's weak compared
# with $\sigma_z$, but because neighbouring stars then phase-mix at different rates, its
# effect on the spiral's fine structure grows much faster than the heating does.
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
# ## Checking the heating rate
#
# Velocity diffusion also heats the population. For an Itô process with velocity
# noise, the mean energy of an ensemble grows at a rate set only by the trace of the
# velocity block of $\mathbf{D}$, in *any* potential:
#
# $$
# \frac{\mathrm{d}\langle E \rangle}{\mathrm{d}t} = \frac{1}{2}\,
# \mathrm{tr}\,\mathbf{D}_{vv} = \frac{1}{2} D_{v_z v_z}.
# $$
#
# We can check that with the snapshots:


# %%
def energy(w):
    return (w.potential_energy(pot) + w.kinetic_energy()).to_value(u.kpc**2 / u.Myr**2)


fig, ax = plt.subplots(figsize=(5, 4))
E0 = energy(w0)
for name, color in zip(["weak diffusion", "strong diffusion"], ["C0", "C1"]):
    dE = [np.mean(energy(w) - E0) for w in snapshots[name]]
    # coefficients() evaluates a model's drift and tensor at given positions
    _, D_model = runs[name].coefficients(w0[:1])
    D_vz = D_model[0, 5, 5]
    ax.plot(snap_times, dE, marker="o", ls="none", color=color, label=name)
    ax.plot(snap_times, 0.5 * D_vz * np.array(snap_times), color=color)
ax.set(xlabel="$t$ [Myr]", ylabel=r"$\langle E(t) - E(0) \rangle$ [(kpc/Myr)$^2$]")
ax.legend()

# %% [markdown]
# The heating is modest: by 600 Myr the strong-diffusion run has gained the energy of
# a random walk of about 7 km/s in $v_z$, less than half of $\sigma_z$. Yet the spiral
# is already badly blurred, because the scatter in vertical *phase* is amplified by
# phase mixing: it grows as
# $D\,t^3$ rather than $D\,t$. Phase spirals are therefore sensitive probes of
# small-scale perturbations to disc orbits.
#
# Here the diffusion is the same everywhere and only affects $v_z$. More realistic
# models, whose diffusion depends on the orbit, are coming in a later release of Gala.
