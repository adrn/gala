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
# In this tutorial, we will demonstrate how to use the mock stream generation functionality in Gala to simulate a stellar stream with a mass-evolving progenitor star cluster, and a non-uniform mass-loss history. For simplicity, we will assume that the scale radius of the progenitor cluster does not change and only its mass evolves from an initial mass of $10^5~{\rm M}_\odot$, losing mass primarily in bursts around pericentric passages.
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
# ## Define the Milky Way potential and progenitor orbit
#
# We'll start by defining a Milky Way potential model using the built-in
# `MilkyWayPotential` class. This provides a simple but reasonable model for
# the Milky Way's mass distribution:

# %%
mw = gp.MilkyWayPotential(version="latest")

# %% [markdown]
# Next, we'll define the initial conditions for our progenitor cluster. We'll
# place it at a position of (13, 0, 20) kpc in Galactocentric Cartesian
# coordinates with a velocity that will put it on an eccentric orbit:

# %%
prog_w0 = gd.PhaseSpacePosition(
    pos=[13.0, 0.0, 20.0] * u.kpc, vel=[0, 130.0, 50] * u.km / u.s
)

# %% [markdown]
# Now we'll integrate the orbit of the progenitor for 6 Gyr using a 1 Myr
# timestep:

# %%
orbit = mw.integrate_orbit(prog_w0, t1=0, t2=6 * u.Gyr, dt=1 * u.Myr)
print(f"Orbit eccentricity: {orbit.eccentricity()}")

# %% [markdown]
# Let's visualize the orbit in 3D:

# %%
_ = orbit.plot()

# %% [markdown]
# We can also look at the progenitor's distance from the Galactic center as a function of time. We expect the cluster to lose more mass around pericenters:

# %%
fig, ax = plt.subplots(figsize=(6, 4), layout="tight")
ax.plot(orbit.t.to_value(u.Gyr), orbit.physicsspherical.r.to_value(u.kpc))

# %% [markdown]
# ## Define a realistic mass-loss history
#
# Now we'll define a mass-loss history for the progenitor cluster that
# concentrates mass loss around pericentric passages. First, let's find all of
# the pericenters along the orbit:

# %%
peri, peri_times = orbit.pericenter(return_times=True, func=None)

# %% [markdown]
# We'll model the mass-loss probability as a mixture of Gaussians centered on each
# pericenter, with a uniform background component to represent steady tidal stripping.
# This therefore represents a scenario where stars are stripped continuously but with
# enhanced stripping at pericenters:

# %%
K = len(peri_times) + 1  # number of mixture components

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

# %% [markdown]
# Now we'll compute the cumulative distribution function (CDF) of the mixture model,
# which we can use to determine how the cluster mass evolves with time:

# %%
t_grid = orbit.t.to_value(u.Myr)
cum_pdf = integrate.cumulative_simpson(y=gmm.pdf(t_grid), x=t_grid)

# %% [markdown]
# Good - this looks like a reasonable mass-loss history. We'll assume the cluster starts
# with an initial mass of $10^5~{\rm M}_\odot$ and loses 90% of its mass over the 6 Gyr
# integration. We create an interpolator for the time-varying mass and visualize the
# mass evolution:

# %%
init_mass = 1e5 * u.Msun
mass_loss_frac = 0.9

prog_mass_t = interpolate.InterpolatedUnivariateSpline(
    0.5 * (t_grid[1:] + t_grid[:-1]), init_mass * (1 - mass_loss_frac * cum_pdf)
)
plt.plot(t_grid, prog_mass_t(t_grid))

# %% [markdown]
# ## Sample particle release times from the mass-loss history
#
# We'll now sample 8000 particle release times from our mass-loss history. This
# will determine when stars are stripped from the progenitor:

# %%
release_times = gmm.sample(shape=8000, rng=np.random.default_rng(12345)) * u.Myr

# %% [markdown]
# Now we'll bin these release times to match our integration timesteps and count
# how many particles should be released at each time step:

# %%
stream_t = np.arange(0.0, 6000.0 + 1e-3, 1) * u.Myr

release_idx = np.digitize(release_times.to_value(u.Myr), stream_t.to_value(u.Myr)) - 1
release_idx, n_release = np.unique(release_idx, return_counts=True)

n_particles = np.zeros(len(stream_t), dtype=int)
n_particles[release_idx] = n_release

# %% [markdown]
# Let's visualize the particle release rate over time:

# %%
plt.plot(stream_t[release_idx], n_release)

# %% [markdown]
# ## Generate the mock stream
#
# Now we're ready to generate the stream simulation. We'll use a `ChenStreamDF`
# distribution function to model the velocity distribution of escaping particles.
# We also need to define a time-varying potential for the progenitor cluster
# using `TimeInterpolatedPotential`, which allows us to specify the mass of the cluster
# at each time knot:

# %%
df = gd.ChenStreamDF()
prog_pot = gp.TimeInterpolatedPotential(
    gp.PlummerPotential,
    time_knots=stream_t,
    m=prog_mass_t(stream_t.to_value(u.Myr)) * u.Msun,
    b=10 * u.pc,
    units="galactic",
)

# %% [markdown]
# With the distribution function, Milky Way potential, and progenitor potential
# all defined, we can now create a `MockStreamGenerator` and generate the stream.
# This will integrate the orbits of all released particles from their release
# times to the present:

# %%
gen = gd.MockStreamGenerator(df, mw, progenitor_potential=prog_pot)
stream, prog_f = gen.run(
    prog_w0,
    prog_mass=prog_mass_t(stream_t.to_value(u.Myr)) * u.Msun,
    t=stream_t,
    n_particles=n_particles,
    Integrator="leapfrog",
)

# %% [markdown]
# ## Generate a comparison stream with constant mass
#
# To see the effect of the time-varying mass, let's also generate a stream with
# a static progenitor mass (equal to the initial mass). This will help us
# see how the mass-loss history affects the stream structure:

# %%
prog_pot_static = gp.PlummerPotential(m=prog_mass_t(0.0), b=10 * u.pc, units="galactic")
gen_static = gd.MockStreamGenerator(df, mw, progenitor_potential=prog_pot_static)
stream_static, prog_f_static = gen_static.run(
    prog_w0, prog_mass=prog_mass_t(0.0) * u.Msun, t=stream_t, Integrator="leapfrog"
)

# %% [markdown]
# ## Visualize the streams
#
# Now let's rotate both streams into the progenitor's orbital plane to better
# see their structure:

# %%
stream_rot = stream.rotate_to_progenitor_plane(prog_f)
stream_rot_static = stream_static.rotate_to_progenitor_plane(prog_f_static)

# %% [markdown]
# Let's plot both streams together to compare them:

# %%
fig = stream_rot.plot(marker="o", s=1, alpha=0.5)
fig = stream_rot_static.plot(marker="o", s=1, alpha=0.5, color="C1", axes=fig.axes)

# %% [markdown]
# We can create a more detailed comparison by plotting 2D histograms of both
# streams side-by-side. This shows the density distribution in the progenitor's
# orbital plane:

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

# %% [markdown]
# Finally, let's compare the linear density profiles along the stream. This
# clearly shows how the realistic mass-loss history (with enhanced stripping at
# pericenters) produces a different density distribution compared to the static
# mass case:

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
