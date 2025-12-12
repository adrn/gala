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

# %% [markdown]
# # What's New in Gala v1.11
#
# Gala v1.11 introduces several new features and improvements, especially in the potential modeling functionality. Below are some highlights of the new features, but see the changelog for the full list of additions and changes.

# %%
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic

# %matplotlib inline

# %% [markdown]
# ---
#
# ## Potential models for arbitrary spherical profiles
#
# The new `SphericalSplinePotential` class allows you to create spherical potential models from tabulated data. You can specify either the density, enclosed mass, or potential values on a grid of radial positions. This is useful when you have a density profile with no closed-form potential, when working with simulation data, or if you want to create a custom potential model that is not implemented in Gala.
#
# For example, imagine we have a density profile with a complex or non-existent closed-form potential solution. Here, we will use a Gaussian-truncated NFW profile as an example:
# $$
# \rho(r) = \frac{\rho_0}{(r/r_s)(1 + r/r_s)^2} \, \exp\left(-\frac{r^2}{r_t^2}\right)
# $$
#
# We can create a `SphericalSplinePotential` from the density profile as follows.
#
# First, implement the density profile:


# %%
def truncated_nfw_density(
    r, rho0=1.6e7 * u.Msun / u.kpc**3, r_s=15 * u.kpc, r_t=100 * u.kpc
):
    uu = r / r_s
    return rho0 / (uu * (1 + uu) ** 2) * np.exp(-((r / r_t) ** 2))


# %% [markdown]
# Evaluate on a grid of radius values. In cases like this, it's good to evaluate on a wide radial range (i.e. larger than the range of radii that you anticipate evaluating on) because the model does not extrapolate beyond the min/max radius values of the defined grid.

# %%
r_grid = np.logspace(-2, 3, 512) * u.kpc

rho0 = 1.6e7 * u.Msun / u.kpc**3
r_s = 15 * u.kpc
r_t = 100 * u.kpc
rho_grid = truncated_nfw_density(r_grid, rho0=rho0, r_s=r_s, r_t=r_t)

# %%
truncated_nfw_pot = gp.SphericalSplinePotential(
    r_knots=r_grid,
    spline_values=rho_grid,
    spline_value_type="density",
    units=galactic,
)

# %% [markdown]
# For comparison, we will also define a standard NFW profile with the same central density and scale radius:

# %%
compare_nfw_pot = gp.NFWPotential(rho0 * 4 * np.pi * r_s**3, r_s, units=galactic)

# %% [markdown]
# Now we can plot and compare the density profiles and, for example, the circular velocity curves derived from the two models:

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")

ax = axes[0]
ax.loglog(
    r_grid.value,
    rho_grid.to_value(u.Msun / u.pc**3),
    marker="",
    label="Truncated NFW density",
)
ax.loglog(
    r_grid.value,
    compare_nfw_pot.density(r=r_grid).to_value(u.Msun / u.pc**3),
    ls="-",
    marker="",
    label="NFW density",
)
ax.axvline(r_t.to_value(u.kpc), ls="-", color="#aaaaaa", zorder=-10)
ax.set(
    xlabel="radius, $r$ [kpc]",
    ylabel=r"density, $\rho(r)$ " + f"[{u.Msun / u.pc**3:latex_inline}]",
    ylim=(1e-15, 2e2),
)
ax.legend(fontsize=14)

# ---

ax = axes[1]
ax.plot(
    r_grid.value,
    truncated_nfw_pot.circular_velocity(r=r_grid).to_value(u.km / u.s),
    marker="",
)
ax.plot(
    r_grid.value,
    compare_nfw_pot.circular_velocity(r=r_grid).to_value(u.km / u.s),
    ls="-",
    marker="",
)
ax.axvline(r_t.to_value(u.kpc), ls="-", color="#aaaaaa", zorder=-10)
ax.set(
    xlabel="radius, $r$ [kpc]",
    ylabel=r"circular velocity, $v_c(r)$ " + f"[{u.km / u.s:latex_inline}]",
)

# %% [markdown]
# ---
#
# ## Potential classes now understand coordinate symmetries
#
# Spherical and axisymmetric potential models can now be evaluated using radius and radius+vertical position, respectively, instead of full 3D Cartesian coordinates. For example, for most spherical potential methods (e.g., `potential()`, `gradient()`, `mass_enclosed()`, `circular_velocity()`, etc.), you can call these functions with `r=...` instead of passing in a Cartesian coordinate array. Similarly, cylindrical potentials can use `R=...` and `z=...` in these same functions. This makes it more convenient to evaluate symmetric potentials along specific axes.

# %%
# Spherical potential example - use r= for spherical radius
hernquist = gp.HernquistPotential(m=1e12 * u.Msun, c=10 * u.kpc, units=galactic)

r = np.linspace(1, 100, 50) * u.kpc

# Old way: construct full 3D coordinates
xyz = np.zeros((3, len(r))) * u.kpc
xyz[0] = r
mass_old_way = hernquist.mass_enclosed(xyz)

# New way: just pass r=
mass_new_way = hernquist.mass_enclosed(r=r)

# %%
# Cylindrical potential example - use R= and z= for cylindrical coordinates
disk = gp.MiyamotoNagaiPotential(
    m=6e10 * u.Msun, a=3 * u.kpc, b=0.28 * u.kpc, units=galactic
)

R = np.linspace(1, 20, 50) * u.kpc

# Evaluate using cylindrical coordinates directly
# For example:
energy_cyl = disk.energy(R=R, z=0.5 * u.kpc)
density_cyl = disk.density(R=R, z=0.5 * u.kpc)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

axes[0].plot(R, energy_cyl)
axes[0].set_xlabel("$R$ [kpc]")
axes[0].set_ylabel(f"energy [{energy_cyl.unit:latex_inline}]")

axes[1].semilogy(R, density_cyl)
axes[1].set_xlabel("$R$ [kpc]")
axes[1].set_ylabel(f"density [{density_cyl.unit:latex_inline}]")

fig.suptitle("Miyamoto-Nagai disk", fontsize=18)

# %% [markdown]
# ---
#
# ## Potential parameters can now be time dependent
#
# The new `TimeInterpolatedPotential` class wraps any potential class to support time-dependent parameters by interpolating their values. This is useful for modeling evolving systems like systems with mass loss, growing potentials, or time-varying pattern speeds (e.g., slowing bars). This works for potential parameters (e.g., mass, scale radius, etc.) along with the potential's origin and rotation.
#
# As a first example, we will create a Plummer potential with a mass that increases non-linearly over time:

# %%
# Define time knots and mass values
t_knots = np.linspace(0, 2, 11) * u.Gyr
mass_knots = (
    1e10 * np.linspace(1, np.sqrt(5), len(t_knots)) ** 2 * u.Msun
)  # mass increasing from 1e10 to 5e10

growing_plummer_pot = gp.TimeInterpolatedPotential(
    gp.PlummerPotential, t_knots, m=mass_knots, b=1 * u.kpc, units=galactic
)

plt.figure(figsize=(6, 4))
plt.plot(t_knots, mass_knots.value / 1e10)
plt.xlabel("$t$ [Gyr]")
plt.ylabel(f"mass [$10^{{10}}$ {mass_knots.unit:latex_inline}]")

# %% [markdown]
# These potentials can be used just like any other Gala potential model. For example, we can integrate an orbit in this potential:

# %%
# Compare orbits in time-varying vs static potential
w0 = gd.PhaseSpacePosition(pos=[5, 0, 0] * u.kpc, vel=[0, 75, 0] * u.km / u.s)

orbit_growing = growing_plummer_pot.integrate_orbit(
    w0, t1=0, t2=2 * u.Gyr, dt=1 * u.Myr
)

# Integrate in static potential (with initial mass)
static_plummer_pot = gp.PlummerPotential(m=mass_knots[0], b=1 * u.kpc, units=galactic)
orbit_static = static_plummer_pot.integrate_orbit(w0, t1=0, t2=2 * u.Gyr, dt=1 * u.Myr)

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")

orbit_growing.plot(
    ["x", "y"], lw=2, axes=axes[0], auto_aspect=False, label="Time-varying mass"
)
orbit_static.plot(
    ["x", "y"], axes=axes[0], auto_aspect=False, label="Static mass", lw=1, alpha=0.5
)
axes[0].legend()
axes[0].set_title("Orbital trajectories")

axes[1].plot(
    orbit_growing.t.to(u.Gyr),
    orbit_growing.spherical.distance.to(u.kpc),
    label="Time-varying",
)
axes[1].plot(
    orbit_static.t.to(u.Gyr), orbit_static.spherical.distance.to(u.kpc), label="Static"
)
axes[1].set_xlabel("Time [Gyr]")
axes[1].set_ylabel("Distance [kpc]")
axes[1].legend()
axes[1].set_title("Radial distance vs time")

# %% [markdown]
# ---
#
# ## String unit systems
#
# Unit systems can now be specified using string names like `'galactic'` or `'dimensionless'` when initializing potentials or replacing units. This is more convenient than importing the unit system objects.

# %%
# Create potentials with string unit systems
pot_galactic = gp.HernquistPotential(m=1e12 * u.Msun, c=10 * u.kpc, units="galactic")
print(f"Units: {pot_galactic.units}")

pot_solar = gp.KeplerPotential(m=1 * u.Msun, units="solarsystem")
print(f"Units: {pot_solar.units}")

# %% [markdown]
# ---
#
# ## String integrator names
#
# Integrators can now be specified using lowercase string names like `'leapfrog'`, `'dopri853'`, or `'ruth4'` instead of importing the integrator classes. This works in `Hamiltonian.integrate_orbit()`, `DirectNBody.integrate_orbit()`, and `MockStreamGenerator.run()`.

# %%
# Define a potential and initial conditions
pot = gp.NFWPotential.from_circular_velocity(
    v_c=200 * u.km / u.s, r_s=15 * u.kpc, units="galactic"
)
H = gp.Hamiltonian(pot)
w0 = gd.PhaseSpacePosition(pos=[10, 0, 0] * u.kpc, vel=[0, 180, 50] * u.km / u.s)

# Integrate with different integrators using string names
orbit_leapfrog = H.integrate_orbit(
    w0, dt=1 * u.Myr, n_steps=1000, Integrator="leapfrog"
)
orbit_ruth4 = H.integrate_orbit(w0, dt=1 * u.Myr, n_steps=1000, Integrator="ruth4")
orbit_dopri = H.integrate_orbit(w0, dt=1 * u.Myr, n_steps=1000, Integrator="dopri853")

# Compare the results
fig, ax = plt.subplots(figsize=(6, 6))
orbit_leapfrog.plot(["x", "y"], axes=ax, label="Leapfrog", alpha=0.7)
orbit_ruth4.plot(["x", "y"], axes=ax, label="Ruth4", alpha=0.7)
orbit_dopri.plot(["x", "y"], axes=ax, label="DOPRI853", alpha=0.7)
ax.legend()
ax.set_title("Orbits with different integrators")

# %% [markdown]
# ---
#
# ## The `MilkyWayPotential` classes have been combined
#
# The `MilkyWayPotential` and `MilkyWayPotential2022` classes have been combined into a single `MilkyWayPotential` class with a `version=` keyword argument. Use `version='v1'` for the original model or `version='v2'` (or `'latest'`) for the 2022 model.

# %%
mw_v1 = gp.MilkyWayPotential(version="v1")  # formerly: MilkyWayPotential()
mw_v2 = gp.MilkyWayPotential(version="v2")  # formerly: MilkyWayPotential2022()
mw_latest = gp.MilkyWayPotential(version="latest")  # same as v2

# %% [markdown]
# ---
#
# ## Mock streams can now be generated using the Leapfrog Integrator
#
# The Leapfrog integrator can now be used with `MockStreamGenerator` by passing `Integrator='leapfrog'` (or the class directly) in the `run()` method. This allows faster stream generation for cases where the adaptive integrator is not needed.

# %%
from gala.dynamics import mockstream as ms

# Set up a Milky Way potential and progenitor
mw = gp.MilkyWayPotential(version="v2")
prog_w0 = gd.PhaseSpacePosition(pos=[15, 0, 0] * u.kpc, vel=[0, 180, 50] * u.km / u.s)

# Define the progenitor mass and stream distribution function
prog_pot = gp.PlummerPotential(m=1e5 * u.Msun, b=10 * u.pc, units="galactic")
df = ms.ChenStreamDF()

# Create the mock stream generator
gen = ms.MockStreamGenerator(df=df, hamiltonian=mw, progenitor_potential=prog_pot)

# Generate stream using the Leapfrog integrator
stream, prog = gen.run(
    prog_w0,
    t1=0 * u.Gyr,
    t2=-2 * u.Gyr,
    dt=-1 * u.Myr,
    prog_mass=prog_pot.parameters["m"],
    n_particles=1,  # 1 particle per release time (leading and trailing)
    Integrator="leapfrog",  # Use string name for integrator!
)

print(f"Generated stream with {stream.shape[0]} particles")

# %%
_ = stream.plot()
