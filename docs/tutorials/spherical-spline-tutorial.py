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
# # Building custom spherical potential models with spline interpolation
#
# In this tutorial, we will demonstrate how to use the `SphericalSplinePotential` class to construct flexible, spherically-symmetric potential models from tabulated data. This is useful when you have a density profile with no simple closed-form potential, or when you want to approximate a potential from simulation data.
#
# We will explore these two use cases:
#
# 1. **Analytic density with no closed-form potential**: You have an analytic density profile for which the potential cannot be expressed in closed form.
# 2. **Simulation-based density profile**: You have density measurements from a simulation and want to quickly generate an approximate potential model (e.g., for orbit integration or dynamical analysis).
#
# ### Notebook Setup and Package Imports

# %%
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic

# %% [markdown]
# ## Use Case 1: Analytic Density with No Closed-Form Potential
#
# ### The Problem
#
# Suppose we are studying a stellar halo component with a density profile that falls off as a power law at intermediate radii but has a steeper break at large radii. We want to model this with a density function:
#
# $$
# \rho(r) = \frac{\rho_0}{(1 + r/r_s)^{\alpha}} \exp\left(-\left(\frac{r}{r_{\rm cut}}\right)^2\right)
# $$
#
# where $\rho_0$ is a normalization, $r_s$ is a scale radius, $\alpha$ controls the inner power-law slope, and $r_{\rm cut}$ is an exponential cutoff radius. For arbitrary $\alpha$ and the exponential cutoff, there is no simple closed-form expression for the potential.
#
# ### Define the Density Profile
#
# Let's define this density profile in Python:


# %%
def halo_density(
    r, rho0=1e9 * u.Msun / u.kpc**3, rs=10.0 * u.kpc, alpha=2.5, rcut=100.0 * u.kpc
):
    return rho0 / (1 + r / rs) ** alpha * np.exp(-((r / rcut) ** 2))


# %% [markdown]
# Let's visualize this density profile over a range of radii:

# %%
r_grid = np.geomspace(0.1, 250, 512) * u.kpc
rho_grid = halo_density(r_grid)

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(r_grid, rho_grid)
ax.set_xlabel("$r$ [kpc]")
ax.set_ylabel(r"$\rho(r)$ [$M_\odot\,{\rm kpc}^{-3}$]")
ax.set_title("Stellar Halo Density Profile")
ax.grid(True, alpha=0.3)

# %% [markdown]
# ### Build a SphericalSplinePotential from the Density
#
# To construct a potential model, we need to:
# 1. Choose radial knot locations where we will sample the density
# 2. Evaluate the density at those knots
# 3. Create a `SphericalSplinePotential` with `spline_value_type='density'`
#
# **Some notes to consider:**
# - Use enough knots to capture the curvature of the density profile
# - Extend the knot grid slightly beyond the region of interest to avoid edge effects from the spline boundary conditions
# - Use logarithmic spacing for knots since the density varies over many orders of magnitude

# %%
# Define radial knots - we use log spacing and extend beyond the region of interest
r_knots = np.geomspace(1e-2, 500, 512) * u.kpc

# Evaluate density at knots
rho_knots = halo_density(r_knots)

# Create the spline potential
halo_pot = gp.SphericalSplinePotential(
    r_knots=r_knots,
    spline_values=rho_knots,
    spline_value_type="density",
    interpolation_method="cspline",
    units=galactic,
)

# %% [markdown]
# Now we can use this potential like any other Gala potential! Let's visualize the potential and the density recovered from the spline:

# %%
r_eval = np.geomspace(0.1, 250, 256) * u.kpc
xyz_eval = np.zeros((3, len(r_eval)))
xyz_eval[0] = r_eval.value
xyz_eval = xyz_eval * r_eval.unit

# Evaluate potential and density
phi_eval = halo_pot.energy(xyz_eval)
rho_recovered = halo_pot.density(xyz_eval)

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# Plot potential
axes[0].plot(r_eval, phi_eval)
axes[0].set_ylabel(rf"$\Phi$ [{phi_eval.unit}]")
axes[0].grid(True, alpha=0.3)
axes[0].set_xscale("log")

# Plot density comparison
axes[1].loglog(r_eval, rho_recovered, label="Recovered from spline", alpha=0.8)
axes[1].loglog(r_eval, halo_density(r_eval), "--", label="Original density", alpha=0.6)
axes[1].scatter(
    r_knots, rho_knots, s=1, c="C2", alpha=0.4, label="Knot locations", zorder=10
)
axes[1].set_xlabel("$r$ [kpc]")
axes[1].set_ylabel(r"$\rho(r)$ [$M_\odot\,{\rm kpc}^{-3}$]")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()

# %% [markdown]
# The recovered density matches the input density extremely well! Let's compute some orbits in this potential.
#
# ### Computing Orbits in the Custom Potential
#
# We can integrate orbits in this custom potential just as we would with any built-in potential. Let's launch test particles from different radii and compare their circular velocities:

# %%
# Compute circular velocity curve
r_circ = np.linspace(1, 100, 200) * u.kpc
xyz_circ = np.zeros((3, len(r_circ)))
xyz_circ[0] = r_circ.value
xyz_circ = xyz_circ * r_circ.unit

v_circ = halo_pot.circular_velocity(xyz_circ)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(r_circ, v_circ)
ax.set_xlabel("$r$ [kpc]")
ax.set_ylabel(r"$v_{\rm circ}$ [km/s]")
ax.set_title("Circular Velocity Curve")
ax.grid(True, alpha=0.3)

# %% [markdown]
# Now let's integrate some orbits with different initial conditions:

# %%
# Define initial conditions: particles at different radii with near-circular velocities
r_init = np.array([10.0, 30.0, 50.0, 75]) * u.kpc
v_init = 0.9 * halo_pot.circular_velocity(
    [r_init.value, np.zeros(r_init.size), np.zeros(r_init.size)] * r_init.unit
)

# Create initial phase-space positions
w0_list = []
for r, v in zip(r_init, v_init):
    w0_list.append(
        gd.PhaseSpacePosition(
            pos=[r.value, 0, 0] * r.unit,
            vel=[0, v.value, 0] * v.unit,
        )
    )

# Integrate orbits
w0s = gd.combine(w0_list)
orbits = halo_pot.integrate_orbit(w0s, dt=1 * u.Myr, t1=0, t2=2 * u.Gyr)

# %% [markdown]
# Visualize the orbits:

# %%
fig = orbits.plot(["x", "y"])

# %% [markdown]
# ## Use Case 2: Simulation-Based Density Profile
#
# ### The Problem
#
# Suppose you have a cosmological simulation and measured the spherically-averaged density profile of a dark matter halo. You want to create a smooth potential model for this halo to compute orbits of satellites, calculate dynamical quantities like circular velocities, etc.
#
# ### Generate Mock Simulation Data
#
# For this example, we'll generate mock "simulation" data that might come from radial binning of particle positions. Real simulation data would be read from a file, but the process is the same:

# %%
# Simulate measuring density in radial bins (e.g., from N-body particles)
rng = np.random.default_rng(42)


# "True" density profile (NFW-like with some scatter)
def nfw_density(r, rho_s=1e8 * u.Msun / u.kpc**3, r_s=15.0 * u.kpc):
    """NFW density profile"""
    x = r / r_s
    return rho_s / (x * (1 + x) ** 2)


# Radial bins (as might come from a simulation)
r_bins = np.logspace(-0.5, 2.2, 30) * u.kpc

# "Measured" densities with some noise to simulate finite sampling
mnfw = 1e12 * u.Msun
rs = 15 * u.kpc
rho0 = mnfw / (4 * np.pi * rs**3)
rho_bins = nfw_density(r_bins, rho_s=rho0, r_s=rs)
# Add log-normal scatter to simulate measurement uncertainty
scatter = rng.lognormal(mean=0, sigma=0.1, size=len(r_bins))
rho_bins_measured = rho_bins * scatter

# %% [markdown]
# Let's visualize our "measured" simulation data:

# %%
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the "true" profile
r_true = np.logspace(-1, 2.5, 200) * u.kpc
ax.loglog(
    r_true,
    nfw_density(r_true, rho0, rs),
    "k--",
    alpha=0.3,
    label="True NFW profile",
    lw=2,
)

# Plot the "measured" data points
ax.loglog(r_bins, rho_bins_measured, "o", label="Simulated measurements", ms=6)

ax.set_xlabel("$r$ [kpc]")
ax.set_ylabel(r"$\rho(r)$ [$M_\odot\,{\rm kpc}^{-3}$]")
ax.set_title("Dark Matter Halo Density from Simulation")
ax.legend()
ax.grid(True, alpha=0.3)

# %% [markdown]
# ### Build a Potential from Simulation Data
#
# Now we'll create a `SphericalSplinePotential` directly from these "measured" densities. The cubic spline interpolation will smooth over the measurement noise:

# %%
# Create potential from simulation data
sim_pot = gp.SphericalSplinePotential(
    r_knots=r_bins,
    spline_values=rho_bins_measured,
    spline_value_type="density",
    interpolation_method="cspline",
    units=galactic,
)

# %% [markdown]
# Let's compare the rotation curve from our spline potential to what we would get from the true NFW profile:

# %%
# For comparison, create an NFW potential with the same parameters
nfw_pot = gp.NFWPotential(m=mnfw, r_s=rs, units=galactic)

# Compute circular velocities
r_test = np.linspace(1, 100, 200) * u.kpc
xyz_test = np.zeros((3, len(r_test)))
xyz_test[0] = r_test.value
xyz_test = xyz_test * r_test.unit

v_circ_sim = sim_pot.circular_velocity(xyz_test)
v_circ_nfw = nfw_pot.circular_velocity(xyz_test)

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True, layout="tight")

# Circular velocity comparison
axes[0].plot(r_test, v_circ_sim, label="Spline potential (from sim data)", lw=2)
axes[0].plot(r_test, v_circ_nfw, "--", label="True NFW potential", alpha=0.7, lw=2)
axes[0].set_ylabel(r"$v_{\rm circ}$ [km/s]")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Density comparison
rho_sim = sim_pot.density(xyz_test)
rho_nfw_true = nfw_density(r_test, rho_s=rho0, r_s=rs)

axes[1].loglog(r_test, rho_sim, label="Spline (smoothed sim data)", lw=2)
axes[1].loglog(r_test, rho_nfw_true, "--", label="True NFW", alpha=0.7, lw=2)
axes[1].scatter(
    r_bins,
    rho_bins_measured,
    s=20,
    c="gray",
    alpha=0.5,
    label="Sim measurements",
    zorder=10,
)
axes[1].set_xlabel("$r$ [kpc]")
axes[1].set_ylabel(r"$\rho(r)$ [$M_\odot\,{\rm kpc}^{-3}$]")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# %% [markdown]
# Excellent! The spline potential reproduces the true NFW profile well, even though we only provided noisy binned measurements. The spline interpolation naturally smooths over the noise.
#
# ### Practical Considerations for Simulation Data
#
# When working with simulation data:
#
# 1. **Choose appropriate radial bins**: Use bins that are finely spaced where the density varies rapidly (typically at small radii) and coarser bins at large radii.
#
# 2. **Handle measurement uncertainties**: If your measurements have large uncertainties, consider:
#    - Using `interpolation_method='akima'` or `'steffen'` to reduce overshoot between noisy points
#    - Smoothing the data before creating the spline
#    - Using more radial bins if you have sufficient particle counts
#
# 3. **Extend beyond the region of interest**: Include radial bins beyond the maximum radius where you plan to integrate orbits. This prevents edge effects in the spline.
#
# 4. **Validate the result**: Always plot the recovered density, potential, and circular velocity to ensure the spline is behaving sensibly.

# %% [markdown]
# ### Exercise: Comparing Interpolation Methods
#
# Different interpolation methods can produce different results, especially with noisy data. Let's compare cubic spline vs. Akima interpolation on our simulated data:

# %%
# Create potentials with different interpolation methods
sim_pot_akima = gp.SphericalSplinePotential(
    r_knots=r_bins,
    spline_values=rho_bins_measured,
    spline_value_type="density",
    interpolation_method="akima",
    units=galactic,
)

# Compare circular velocities
v_circ_akima = sim_pot_akima.circular_velocity(xyz_test)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(r_test, v_circ_sim, label="cspline (default)", lw=2)
ax.plot(r_test, v_circ_akima, "--", label="akima", alpha=0.8, lw=2)
ax.plot(r_test, v_circ_nfw, ":", label="True NFW", alpha=0.5, lw=2, color="k")
ax.set_xlabel("$r$ [kpc]")
ax.set_ylabel(r"$v_{\rm circ}$ [km/s]")
ax.set_title("Effect of Interpolation Method")
ax.legend()
ax.grid(True, alpha=0.3)

# %% [markdown]
# Both methods work well for this relatively smooth data. The Akima method tends to produce less overshoot in regions with rapid changes or noise, while cubic splines generally produce smoother second derivatives (which affects the recovered density).

# %% [markdown]
# ## Summary and Recommendations
#
# We've demonstrated how to use `SphericalSplinePotential` for two use cases.
#
# - **Use `'cspline'` interpolation** (the default) when possible, as it provides continuous second derivatives and physically smooth densities.
#
# - **Extend knots beyond your region of interest** by 20% or more to minimize edge effects from boundary conditions. Cubic splines force the second derivative to zero at endpoints, which can create artifacts.
#
# - **Use logarithmic spacing for knots** when working with profiles that span many orders of magnitude in radius or density.
#
# - **Validate your potential**: Always plot the recovered density, potential, circular velocity, and compare to your expectations or input data.
#
# - **For noisy data**, consider using `'akima'` or `'steffen'` interpolation to reduce overshoot, or smooth your input data before creating the spline.

# %% [markdown]
# ### Exercise: Build Your Own Model
#
# Try creating a `SphericalSplinePotential` for a different density profile:
#
# - **Broken power law**: $\rho(r) = \rho_0 \times (r/r_b)^{-\alpha_{\rm in}}$ for $r < r_b$ and $(r/r_b)^{-\alpha_{\rm out}}$ for $r > r_b$
# - **Double power law (Hernquist-like)**: $\rho(r) = \rho_0 / [(r/r_s)^{\gamma}(1 + r/r_s)^{4-\gamma}]$
# - **Your own custom profile**: Combine multiple components or use a functional form from a paper
#
# Compute orbits in your custom potential and visualize the results!
