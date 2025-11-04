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

# %%
# %run nb_setup

# %%
# %matplotlib inline

# %% [markdown]
# # Integrating orbits in a barred galaxy potential
#
# In this tutorial, we'll explore how to integrate orbits in a time-dependent, barred galaxy potential model. In barred models, the bar rotates, so we need to account
# for this time-dependence. There are two ways to do this in Gala:
#
# 1. Rotating frame: Use a static bar potential in a rotating reference frame
# 2. Inertial frame: Use a time-dependent rotating bar potential in an inertial frame
#
# We'll demonstrate both approaches and show that they give similar results, but with different accuracies.
#
# ### Notebook Setup and Package Imports

# %%
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so
from scipy.spatial.transform import Rotation

import gala.dynamics as gd
import gala.potential as gp

# %% [markdown]
# ## Define the bar rotation parameters
#
# We'll set up a bar rotating with a pattern speed of 30 km/s/kpc, which is converted
# to radians per Gyr using `astropy.units`.

# %%
# Set bar pattern speed
with u.set_enabled_equivalencies(u.dimensionless_angles()):
    Omega = 30 * u.km / u.s / u.kpc
    Omega = Omega.to(u.rad / u.Gyr)


# %%
# Create time array spanning ~5 Gyr with 200 steps per rotation period
dt = 2 * np.pi * u.rad / Omega / 200
time_knots = np.arange(0, 5, dt.to(u.Gyr).value) * u.Gyr


# %% [markdown]
# ## Method 1: Static bar in rotating frame
#
# In this approach, we define a static bar potential and integrate orbits in a
# rotating reference frame. The bar remains fixed in this frame, but the frame
# itself rotates with the bar's pattern speed.
#
# ### Create the potential
#
# We'll use a simple, analytic representation of the potential from a Galactic bar and integrate an orbit in the rotating frame of the bar. The total potential model will be a four-component model consisting of the bar, using the [Long & Murali 1992](http://adsabs.harvard.edu/abs/1992ApJ...397...44L) model, plus disk, halo, and nucleus components from the Gala `MilkyWayPotential`. We adjust the disk and bar mass to roughly match the circular velocity of the Milky Way at the solar radius.

# %%
# Use the latest Milky Way potential as a base
mw = gp.MilkyWayPotential(version="latest")

# Create a composite potential with a static bar
bar_mw_static = gp.CCompositePotential()
bar_mw_static["bar"] = gp.LongMuraliBarPotential(
    m=1e10 * u.Msun,
    a=4 * u.kpc,
    b=0.8 * u.kpc,
    c=0.25 * u.kpc,
    alpha=25 * u.deg,
    units="galactic",
)
bar_mw_static["disk"] = mw["disk"].replicate(m=4.1e10 * u.Msun)
bar_mw_static["halo"] = mw["halo"]
bar_mw_static["nucleus"] = mw["nucleus"]

# %% [markdown]
# Let's visualize the isopotential contours of the potential in the x-y plane to
# see the bar perturbation::

# %%
fig, ax = plt.subplots(figsize=(5, 5), layout="tight")

grid = np.linspace(-12, 12, 128)
_ = bar_mw_static.plot_contours(grid=(grid, grid, 0), ax=ax)
ax.set_xlabel("$x$ [kpc]")
ax.set_ylabel("$y$ [kpc]")
ax.set_title("Bar potential")
plt.show()

# %% [markdown]
# We assume that the bar rotates around the z-axis so that the frequency vector is  $\boldsymbol{\Omega} = (0, 0, 1) \times \Omega$. We'll create a `Hamiltonian` object with a `ConstantRotatingFrame` with this frequency:

# %%
frame = gp.ConstantRotatingFrame(Omega=Omega * [0, 0, 1], units="galactic")
H_rotating = gp.Hamiltonian(potential=bar_mw_static, frame=frame)


# %% [markdown]
# To get a set of initial conditions to compute an orbit, we numerically find the co-rotation radius in this potential and integrate an orbit near co-rotation.:


# %%
def find_corotation(potential, Omega_bar):
    """Find the corotation radius numerically."""

    def func(r):
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            v_circ = potential.circular_velocity([r[0], 0, 0] * u.kpc)[0]
            Om = v_circ / (r[0] * u.kpc)
            return (Om - Omega_bar).to(Omega_bar.unit).value ** 2

    res = so.minimize(func, x0=10.0, method="powell")
    return res.x[0] * u.kpc


r_corot = find_corotation(bar_mw_static, Omega)
v_circ = Omega * r_corot


with u.set_enabled_equivalencies(u.dimensionless_angles()):
    pass

# initial conditions at corotation radius
w0 = gd.PhaseSpacePosition(
    pos=[r_corot.value, 0, 0] * r_corot.unit,
    vel=[0, v_circ.value, 0.0] * v_circ.unit,
)

# %% [markdown]
# We can now compute an orbit from these initial conditions in the rotating frame Hamiltonian. We'll integrate the orbit for 5 Gyr with a time step of 1 Myr and then visualize the orbit in the rotating frame:

# %%
# Integration parameters
integrator_kwargs = {"atol": 1e-14, "rtol": 1e-14}
t1 = time_knots.min()
t2 = time_knots.max()
dt_integrate = 0.1 * u.Myr

# %%
orbit_rotating = H_rotating.integrate_orbit(
    w0,
    t1=t1,
    t2=t2,
    dt=dt_integrate,
    Integrator="dopri853",
    Integrator_kwargs=integrator_kwargs,
)

# %%
fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
orbit_rotating.plot(["x", "y"], axes=ax, marker="", linestyle="-", lw=0.5)
_ = ax.set(
    xlim=(-12, 12),
    ylim=(-12, 12),
    xlabel="$x$ [kpc]",
    ylabel="$y$ [kpc]",
    title="Orbit in rotating frame",
)

# %% [markdown]
# This is an orbit circulation around the Lagrange point L5!
#
# We can also visualize the orbit in the inertial frame using the `to_frame()` method:

# %%
orbit_rotating_inertial = orbit_rotating.to_frame(gp.StaticFrame(units="galactic"))

fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
orbit_rotating_inertial.plot(["x", "y"], axes=ax, marker="", linestyle="-", lw=0.5)
_ = ax.set(
    xlim=(-12, 12),
    ylim=(-12, 12),
    xlabel="$x$ [kpc]",
    ylabel="$y$ [kpc]",
    title="Orbit in inertial frame",
)

# %% [markdown]
# ## Method 2: Time-dependent bar in inertial frame
#
# As an alternate approach, we could instead work in the inertial frame and define a time-dependent potential that represents the rotating bar. To do this, we use the `TimeDependentPotential` class in Gala and specify a series of time values and corresponding rotation matrices that describe the bar's orientation at each time. Gala will then interpolate the angles as needed during the orbit integration.
#
#
# ### Create the time-dependent potential
#
# We construct the potential in a very similar way to before, but now the bar component is wrapped in a `TimeInterpolatedPotential` that takes the pre-computed rotation matrices. First, we need to compute the rotation matrices:

# %%
# Pre-compute rotation matrices for each time step
# Negative angle to be comparable to having a frame rotating at +Omega
bar_angle = (-Omega * time_knots).to_value(u.rad)
Rs = Rotation.from_euler("z", bar_angle).as_matrix()


# %% [markdown]
# Now we can construct the full time-dependent potential:

# %%
bar_mw_timedep = gp.CCompositePotential()
bar_mw_timedep["bar"] = gp.TimeInterpolatedPotential(
    gp.LongMuraliBarPotential,
    time_knots=time_knots,
    m=1e10 * u.Msun,
    a=4 * u.kpc,
    b=0.8 * u.kpc,
    c=0.25 * u.kpc,
    units="galactic",
    alpha=25 * u.deg,
    R=Rs,
)
bar_mw_timedep["disk"] = mw["disk"].replicate(m=4.1e10 * u.Msun)
bar_mw_timedep["halo"] = mw["halo"]
bar_mw_timedep["nucleus"] = mw["nucleus"]

# %% [markdown]
# As a sanity check, let's visualize the potential at different times to see the bar rotating:

# %%
fig, axes = plt.subplots(
    1, 3, figsize=(15, 5), layout="constrained", sharex=True, sharey=True
)

times = [0, 10, 20] * u.Myr
for ax, t in zip(axes, times):
    _ = bar_mw_timedep.plot_contours(grid=(grid, grid, 0), t=t, ax=ax)
    ax.set_xlabel("$x$ [kpc]")
    ax.set_title(f"$t = {t.value:.1f}$ Gyr")
    ax.set_aspect("equal")

axes[0].set_ylabel("$y$ [kpc]")

fig.suptitle("Time-dependent bar in inertial frame", fontsize=22)

# %% [markdown]
# Now let's integrate the same orbit as before, but now in the inertial frame with the time-dependent potential. We'll use the same initial conditions and integration parameters as before:

# %%
orbit_inertial = bar_mw_timedep.integrate_orbit(
    w0,
    t1=t1,
    t2=t2,
    dt=dt_integrate,
    Integrator="dopri853",
    Integrator_kwargs=integrator_kwargs,
)

# %%
fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
orbit_inertial.plot(["x", "y"], axes=ax, marker="", linestyle="-", lw=0.5)
_ = ax.set(
    xlim=(-12, 12),
    ylim=(-12, 12),
    xlabel="$x$ [kpc]",
    ylabel="$y$ [kpc]",
    title="Orbit 2 in inertial frame",
)

# %% [markdown]
# Now we can transform the orbit computed in the inertial frame to the rotating frame for comparison:

# %%
orbit_inertial_rotating = orbit_inertial.to_frame(H_rotating.frame)

# %%
fig, axes = plt.subplots(
    1, 2, figsize=(10, 5), layout="constrained", sharex=True, sharey=True
)

orbit_rotating.plot(["x", "y"], axes=axes[0], marker="", linestyle="-", lw=0.5)
_ = axes[0].set(
    xlim=(-12, 12),
    ylim=(-12, 12),
    xlabel="$x$ [kpc]",
    ylabel="$y$ [kpc]",
    title="Orbit computed in rotating frame",
)

orbit_inertial_rotating.plot(
    ["x", "y"],
    axes=axes[1],
    marker="",
    linestyle="-",
    lw=0.5,
)
_ = axes[1].set(
    xlabel="$x$ [kpc]",
    title="Orbit computed in inertial frame",
)

# %% [markdown]
# Excellent, the orbits look very similar! This is expected: they represent the same physical motion, just computed in different ways. Let's quantify how well they match.

# %% [markdown]
# ## Energy conservation: Jacobi integral
#
# In the rotating frame, the relevant conserved quantity is the **Jacobi integral** (also
# called the Jacobi constant or Jacobi energy), not the total energy. Let's examine
# how well each method conserves the Jacobi integral.
#
# The Jacobi integral is:
# $$E_J = E - \mathbf{\Omega} \cdot \mathbf{L}$$
#
# where $E$ is the total energy in the rotating frame and $\mathbf{L}$ is the angular momentum.

# %%
# Compute Jacobi integral for rotating frame orbit
E_rotating = H_rotating.energy(orbit_rotating)

# Compute Jacobi integral for inertial orbit (transformed to rotating frame)
E_inertial_transformed = H_rotating.energy(orbit_inertial_rotating)

# %% [markdown]
# Let's see how well each method conserves the Jacobi integral over the course of the integration:

# %%
# Compute fractional energy conservation
frac_dE_rotating = np.abs((E_rotating[1:] - E_rotating[0]) / E_rotating[0])
frac_dE_inertial = np.abs(
    (E_inertial_transformed[1:] - E_inertial_transformed[0]) / E_inertial_transformed[0]
)


# %%
fig, ax = plt.subplots(figsize=(10, 6), layout="tight")

ax.loglog(
    orbit_rotating.t[1:].to(u.Gyr).value,
    frac_dE_rotating,
    label="Method 1: Rotating frame",
    alpha=0.7,
)
ax.loglog(
    orbit_inertial.t[1:].to(u.Gyr).value,
    frac_dE_inertial,
    label="Method 2: Inertial frame",
    alpha=0.7,
)

ax.axhline(1e-14, color="k", linestyle="--", alpha=0.3, label="Tolerance (rtol)")
ax.set_xlabel("Time [Gyr]")
ax.set_ylabel("Fractional Jacobi integral error $|\\Delta E_J / E_J|$")
ax.set_title("Jacobi Integral Conservation")
ax.legend()
ax.grid(alpha=0.3)
plt.show()

# %% [markdown]
# Both methods conserve the Jacobi integral well, but the rotating frame approach (Method 1 above) generally provides better long-term stability for orbits in a barred potential.
