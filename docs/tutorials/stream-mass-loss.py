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
# %run nb_setup

# %%
# %matplotlib inline

# %% [markdown]
# # Generate a mock stellar stream with a realistic mass-loss history
#
# In this tutorial, we will demonstrate how to use the mock stream generation functionality in Gala to simulate a stellar stream with a mass-evolving progenitor star cluster. For simplicity, we will assume that the scale radius of the progenitor cluster does not change and only its mass evolves from an initial mass of $10^5~{\rm M}_\odot$, losing mass primarily in bursts around pericentric passages.
#
#
#
# ### Notebook Setup and Package Imports

# %%
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, interpolate, stats

import gala.dynamics as gd
import gala.potential as gp

# %% [markdown]
# ## TODO
#
# Blah blah

# %%
mw = gp.MilkyWayPotential(version="latest")

# %%
prog_w0 = gd.PhaseSpacePosition(
    pos=[13.0, 0.0, 20.0] * u.kpc, vel=[0, 130.0, 50] * u.km / u.s
)

# %%
orbit = mw.integrate_orbit(prog_w0, t1=0, t2=6 * u.Gyr, dt=1 * u.Myr)

# %%
_ = orbit.plot()

# %%
fig, ax = plt.subplots(figsize=(6, 4), layout="tight")
ax.plot(orbit.t.to_value(u.Gyr), orbit.physicsspherical.r.to_value(u.kpc))

# %%
peri, peri_times = orbit.pericenter(return_times=True, func=None)

# %%
K = len(peri_times) + 1

t_lim = (orbit.t.to_value(u.Myr).min(), orbit.t.to_value(u.Myr).max())
weights = np.zeros(K)
weights[0] = 1.0
weights[1:] = 0.2
weights /= weights.sum()

gmm = stats.Mixture(
    [stats.Uniform(a=t_lim[0], b=t_lim[1])]
    + [stats.Normal(mu=tt, sigma=25.0) for tt in peri_times.to_value(u.Myr)],
    weights=weights,
)

# %%
t_grid = orbit.t.to_value(u.Myr)
cum_pdf = integrate.cumulative_simpson(y=gmm.pdf(t_grid), x=t_grid)

# %%
init_mass = 1e5 * u.Msun
mass_loss_frac = 0.9

prog_mass_t = interpolate.InterpolatedUnivariateSpline(
    0.5 * (t_grid[1:] + t_grid[:-1]), init_mass * (1 - mass_loss_frac * cum_pdf)
)
plt.plot(t_grid, prog_mass_t(t_grid))

# %%
release_times = gmm.sample(shape=8000, rng=np.random.default_rng(12345)) * u.Myr

# %%
stream_t = np.arange(0.0, 6000.0 + 1e-3, 1) * u.Myr

release_idx = np.digitize(release_times.to_value(u.Myr), stream_t.to_value(u.Myr)) - 1
release_idx, n_release = np.unique(release_idx, return_counts=True)

n_particles = np.zeros(len(stream_t), dtype=int)
n_particles[release_idx] = n_release

# %%
plt.plot(stream_t[release_idx], n_release)

# %%
df = gd.ChenStreamDF()
prog_pot = gp.TimeInterpolatedPotential(
    gp.PlummerPotential,
    time_knots=stream_t,
    m=prog_mass_t(stream_t.to_value(u.Myr)) * u.Msun,
    b=10 * u.pc,
    units="galactic",
)

# %%
gen = gd.MockStreamGenerator(df, mw, progenitor_potential=prog_pot)
stream, prog_f = gen.run(
    prog_w0,
    prog_mass=prog_mass_t(stream_t.to_value(u.Myr)) * u.Msun,
    t=stream_t,
    n_particles=n_particles,
    Integrator="leapfrog",
)

# %%
prog_pot_static = gp.PlummerPotential(m=prog_mass_t(0.0), b=10 * u.pc, units="galactic")
gen_static = gd.MockStreamGenerator(df, mw, progenitor_potential=prog_pot_static)
stream_static, prog_f_static = gen_static.run(
    prog_w0, prog_mass=prog_mass_t(0.0) * u.Msun, t=stream_t, Integrator="leapfrog"
)

# %%
stream_rot = stream.rotate_to_progenitor_plane(prog_f)
stream_rot_static = stream_static.rotate_to_progenitor_plane(prog_f_static)

# %%
fig = stream_rot.plot(marker="o", s=1, alpha=0.5)
fig = stream_rot_static.plot(marker="o", s=1, alpha=0.5, color="C1", axes=fig.axes)

# %%
binsx = np.arange(-10, 10 + 1e-3, 0.05)
binsy = np.arange(-3, 3 + 1e-3, 0.05)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="tight")

ax = axes[0]
for ax, _stream in zip(axes, [stream_rot, stream_rot_static]):
    H, xe, ye = np.histogram2d(
        _stream.x.to_value(u.kpc), _stream.y.to_value(u.kpc), bins=(binsx, binsy)
    )
    ax.pcolormesh(
        xe, ye, H.T, shading="auto", norm=mpl.colors.LogNorm(0.5, 1e2), cmap="magma_r"
    )
    ax.set_xlabel("progenitor frame $x$ [kpc]")

axes[0].set_title("With cluster mass loss and mass-loss history")
axes[1].set_title("No time dependence")
axes[0].set_ylabel("progenitor frame $y$ [kpc]")

# %%
bins = np.linspace(-10, 10, 128)

fig, ax = plt.subplots()
ax.hist(
    stream_rot.x.to_value(u.kpc),
    bins=bins,
    density=True,
    histtype="step",
    label="With cluster mass loss and mass-loss history",
)
ax.hist(
    stream_rot_static.x.to_value(u.kpc),
    bins=bins,
    color="C1",
    density=True,
    histtype="step",
    label="No time dependence",
)
ax.set(
    xlabel="progenitor frame $x$ [kpc]",
    ylabel="linear density",
)
ax.legend(loc="upper left", fontsize=10)

# %%
