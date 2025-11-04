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
# # Working with time-evolving potential models
#
# In this tutorial, we will demonstrate how to use the `TimeInterpolatedPotential`
# wrapper class in Gala to create potentials with time-varying parameters. This is
# useful for modeling scenarios like:
#
# - Mass loss from a star cluster or satellite galaxy
# - Growing or shrinking potentials (e.g., a forming galaxy)
# - Rotating bar potentials with time-varying pattern speeds
# - etc.
#
# The `TimeInterpolatedPotential` class uses GSL splines to interpolate potential
# parameters, origins, and rotation matrices between discrete time knots. We'll
# explore how to use this class with different potential models, control the
# interpolation method, and understand its behavior at the boundaries of the
# interpolation range.
#
# ### Notebook Setup and Package Imports

# %%
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

import gala.dynamics as gd
import gala.potential as gp

# %% [markdown]
# ## Basic usage: Time-varying mass in a point mass potential
#
# Let's start with a simple example: a point mass (Kepler) potential where the
# mass grows linearly with time. This could represent, for example, a growing
# black hole or the infall of matter onto a central object.
#
# First, we'll define our time knots and the corresponding masses at each knot:

# %%
# Define time knots spanning 1 Gyr
time_knots = np.linspace(0, 1, 11) * u.Gyr

# Mass linearly growing from 1e10 to 2e10 solar masses
masses = np.linspace(1e10, 2e10, len(time_knots)) * u.Msun

# %% [markdown]
# Now we create the `TimeInterpolatedPotential`. The first argument is the
# potential class we want to wrap (not an instance, but the class itself), the
# second argument is the array of time knots, and then we pass the time-varying
# parameters using the parameter names of the underlying potential:

# %%
pot_varying_mass = gp.TimeInterpolatedPotential(
    gp.KeplerPotential, time_knots, m=masses, units="galactic"
)

print(repr(pot_varying_mass))

# %% [markdown]
# Let's visualize how the potential energy changes with time at a fixed position:

# %%
test_position = [10.0, 0.0, 0.0] * u.kpc
test_times = np.linspace(0, 1000, 100) * u.Myr

energies = [pot_varying_mass.energy(test_position, t=t) for t in test_times]

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(test_times.to_value(u.Gyr), [e.value for e in energies])
ax.set_xlabel("Time [Gyr]")
ax.set_ylabel(f"Potential Energy [{energies[0].unit:latex_inline}]")
ax.set_title("Energy at fixed position vs. time")

# %% [markdown]
# As expected, the potential energy becomes more negative (deeper) as the mass
# increases. Now let's integrate an orbit in this time-varying potential and
# compare it to an orbit in a static potential with the initial mass:

# %%
# Initial conditions
w0 = gd.PhaseSpacePosition(pos=[10.0, 0.0, 0.0] * u.kpc, vel=[0, 50.0, 0] * u.km / u.s)

# Integrate in time-varying potential
orbit_varying = pot_varying_mass.integrate_orbit(
    w0, t1=0, t2=1000 * u.Myr, dt=1 * u.Myr
)

# Create static potential with initial mass for comparison
pot_static = gp.KeplerPotential(m=masses[0], units="galactic")
orbit_static = pot_static.integrate_orbit(w0, t1=0, t2=1000 * u.Myr, dt=1 * u.Myr)

# %% [markdown]
# Let's visualize how the orbits differ:

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="tight")

ax = axes[0]
orbit_varying.plot(["x", "y"], axes=ax, color="tab:blue", alpha=0.5, auto_aspect=False)
orbit_static.plot(["x", "y"], axes=ax, color="tab:orange", alpha=0.5, auto_aspect=False)
ax.set_title("Orbital trajectories")

ax = axes[1]
orbit_varying.spherical.plot(
    ["t", "distance"], axes=ax, color="tab:blue", alpha=0.5, label="time-varying"
)
orbit_static.spherical.plot(
    ["t", "distance"], axes=ax, color="tab:orange", alpha=0.5, label="static"
)
ax.legend()
ax.set_title("Radial distance vs. time")

# %% [markdown]
# The orbit in the time-varying potential shows the effect of the increasing mass: the orbital radius decreases over time as the central mass grows stronger.
#
# ## Using different interpolation methods
#
# The `TimeInterpolatedPotential` supports several interpolation methods from GSL.
# Let's compare how different methods interpolate the same data. The available
# methods are:
#
# - `'linear'`: Linear interpolation (requires 2+ knots)
# - `'cspline'`: Cubic spline (requires 3+ knots, default)
# - `'akima'`: Akima spline (requires 5+ knots, avoids unphysical wiggles)
# - `'steffen'`: Steffen spline (requires 3+ knots, guarantees monotonicity)

# %%
# For this example, let's use fewer knots to make differences more visible
sparse_times = np.array([0, 250, 500, 750, 1000]) * u.Myr
sparse_masses = np.array([1e10, 1.3e10, 1.8e10, 1.6e10, 2e10]) * u.Msun

# Create potentials with different interpolation methods
interp_methods = ["linear", "cspline", "akima", "steffen"]
pots = {}

for method in interp_methods:
    pots[method] = gp.TimeInterpolatedPotential(
        gp.KeplerPotential,
        sparse_times,
        m=sparse_masses,
        units="galactic",
        interpolation_method=method,
    )

# %% [markdown]
# Now let's compare how each method interpolates the mass parameter:

# %%
eval_times = np.linspace(0, 1000, 200) * u.Myr
test_pos = [10.0, 0.0, 0.0] * u.kpc

fig, ax = plt.subplots(figsize=(8, 5))

# Plot the mass evolution implied by energy at a fixed position
for method in interp_methods:
    energies = [pots[method].energy(test_pos, t=t) for t in eval_times]
    ax.plot(eval_times.to_value(u.Myr), energies, label=method, lw=1)

# Mark the knot positions
ax.scatter(
    sparse_times.to_value(u.Myr),
    [pots["linear"].energy(test_pos, t=t).value for t in sparse_times],
    color="black",
    s=100,
    zorder=10,
    label="knot positions",
)

ax.set_xlabel("Time [Myr]")
ax.set_ylabel(f"Potential Energy [{energies[0].unit.to_string('latex')}]")
ax.legend()
ax.set_title("Comparison of interpolation methods")
ax.grid(alpha=0.3)

# %% [markdown]
# Notice the differences:
#
# - **Linear**: Piecewise linear, no smoothness between segments
# - **Cspline**: Smooth (continuous second derivative) but can overshoot
# - **Akima**: Smooth and avoids overshooting near sharp changes
# - **Steffen**: Guarantees monotonicity between knots (no spurious oscillations)
#
# For physical applications, `steffen` and `akima` are often useful when you
# want to avoid unphysical oscillations in the interpolated values.
#
# ## Time-varying rotation: Modeling a rotating bar
#
# The `TimeInterpolatedPotential` can also interpolate rotation matrices,
# allowing you to model rotating structures like galactic bars. Let's create a
# bar potential that rotates over time:

# %%
# Time knots for rotation
rot_times = np.linspace(0, 3, 128) * u.Gyr

# Create rotation matrices for bar rotation
Omega = np.pi * u.rad / (100 * u.Myr)
angles = (Omega * rot_times).to_value(u.rad)
rotation_matrices = Rotation.from_euler("z", angles).as_matrix()

# Create a bar potential with time-varying rotation
pot_rotating_bar = gp.TimeInterpolatedPotential(
    gp.LongMuraliBarPotential,
    rot_times,
    m=1e10 * u.Msun,
    a=3 * u.kpc,
    b=1 * u.kpc,
    c=0.5 * u.kpc,
    R=rotation_matrices,
    units="galactic",
)

# %% [markdown]
# Let's test the rotation by evaluating the potential gradient at a fixed point
# in space at different times:

# %%
test_pos = [5.0, 0.0, 0.0] * u.kpc
sample_times = np.array([0, 25, 50, 75, 100]) * u.Myr

print("Gradient components at x = [5, 0, 0] kpc at different times:")

for t in sample_times:
    grad = pot_rotating_bar.gradient(test_pos, t=t)
    print(np.squeeze(grad))

# %% [markdown]
# As the bar rotates, the gradient components change. At t=0, the point is along
# the x-axis and the bar is aligned with x, so we see mainly x-direction force.
# As time progresses and the bar rotates, the force direction changes.
#
# Let's integrate an orbit in this rotating bar potential:

# %%
w0_bar = gd.PhaseSpacePosition(
    pos=[8.0, 0.0, 0.0] * u.kpc, vel=[0, 50.0, 0] * u.km / u.s
)

orbit_rot_bar = pot_rotating_bar.integrate_orbit(
    w0_bar, t1=0, t2=2 * u.Gyr, dt=0.5 * u.Myr, Integrator="dop853"
)

# %% [markdown]
# Visualize the orbit in the rotating bar potential:

# %%
fig = orbit_rot_bar.plot(["x", "y"])

# %% [markdown]
# ## Multiple time-varying parameters
#
# You can have multiple parameters varying with time simultaneously. Let's create
# a Hernquist potential where both the mass and scale radius change:

# %%
time_knots_multi = np.linspace(0, 2000, 21) * u.Myr

# Mass decreases (mass loss)
masses_multi = np.linspace(5e11, 1e11, 21) * u.Msun

# Scale radius expands slightly as mass is lost
scale_radii = np.linspace(5, 8, 21) * u.kpc

pot_multi = gp.TimeInterpolatedPotential(
    gp.HernquistPotential,
    time_knots_multi,
    m=masses_multi,
    c=scale_radii,
    units="galactic",
)

# %% [markdown]
# Let's see how the circular velocity curve changes with time:

# %%
radii = np.linspace(1, 30, 50) * u.kpc
sample_times_multi = np.array([0, 500, 1000, 1500, 2000]) * u.Myr

fig, ax = plt.subplots(figsize=(10, 6))

for t in sample_times_multi:
    v_circs = []
    for r in radii:
        pos = [r.value, 0, 0] * u.kpc
        # v_circ = sqrt(r * |dPhi/dr|)
        grad = pot_multi.gradient(pos, t=t)
        v_circ = np.sqrt(r * np.abs(grad[0]))
        v_circs.append(v_circ.to(u.km / u.s).value)

    ax.plot(radii.to_value(u.kpc), v_circs, label=f"t = {t.value:.0f} Myr")

ax.set_xlabel("Radius [kpc]")
ax.set_ylabel("Circular velocity [km/s]")
ax.legend()
ax.grid(alpha=0.3)
ax.set_title("Time evolution of circular velocity curve")

# %% [markdown]
# The circular velocity decreases over time as both the mass decreases and the
# scale radius increases.
#
# ## Boundary behavior: No extrapolation
#
# An important feature of `TimeInterpolatedPotential` is that it does **not**
# extrapolate beyond the time range defined by the time knots. If you try to
# evaluate the potential or integrate an orbit outside this range, you'll get
# NaN values or an error. Let's demonstrate this:

# %%
# Create a potential with a limited time range
limited_times = np.array([100, 200, 300]) * u.Myr
limited_masses = np.array([1e10, 1.5e10, 2e10]) * u.Msun

pot_limited = gp.TimeInterpolatedPotential(
    gp.KeplerPotential,
    limited_times,
    m=limited_masses,
    units="galactic",
    interpolation_method="cspline",
)

test_pos_bounds = [5.0, 0.0, 0.0] * u.kpc

# Try evaluating at different times
test_times_bounds = np.array([50, 150, 250, 350]) * u.Myr

print("Evaluating potential at different times:")
print(
    f"Time knot range: [{limited_times.min().value}, {limited_times.max().value}] Myr"
)
print()
print(f"{'Time [Myr]':<15} {'Energy':<30} {'Status'}")
print("-" * 60)

for t in test_times_bounds:
    energy = pot_limited.energy(test_pos_bounds, t=t)
    status = "Outside range" if t.value < 100 or t.value > 300 else "Valid"
    print(f"{t.to_value(u.Myr):<15.0f} {energy!s:<30} {status}")

# %% [markdown]
# Notice that evaluations at t=50 Myr (before the first knot) and t=350 Myr
# (after the last knot) return NaN. This is by design - the interpolation is only
# valid within the specified time range.
#
# If you try to integrate an orbit that goes outside the time range, you'll get
# an error:

# %%
w0_bounds = gd.PhaseSpacePosition(
    pos=[10.0, 0.0, 0.0] * u.kpc, vel=[0, 100.0, 0] * u.km / u.s
)

try:
    # This should raise an error because we're trying to integrate from 0 to 400 Myr
    # but the potential is only defined from 100 to 300 Myr
    orbit_bad = pot_limited.integrate_orbit(w0_bounds, t1=0, t2=400, dt=1)
except ValueError as e:
    print("Error caught as expected:")
    print(f"  {e}")
