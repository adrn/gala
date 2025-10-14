Spherical spline–interpolated potentials
========================================

.. _spherical-spline:

The :class:`~gala.potential.potential.SphericalSplinePotential` class provides a
flexible, spherically-symmetric potential model constructed from a 1D radial spline.
Instead of hard-coding an analytic profile, the user supplies values of either potential
energy, density, or enclosed mass on a set of radial knots.

This implementation requires GSL (the GNU Scientific Library) to be available at
build/runtime. If Gala was built without GSL support, this class will not be usable.

Notes
~~~~~

- This potential uses a single class front-end that supports three input "value types"

  - density: The potential energy, mass enclosed, and gradients are computed by
    integrating the density.
  - mass (i.e. enclosed mass): The potential energy is computed by
    numerical integration and the density is obtained directly with derivatives of the
    splines.
  - potential: The enclosed mass, gradient, and density functions are obtained
    directly from spline interpolation.
- Supports all `GSL interpolation methods <https://www.gnu.org/software/gsl/doc/html/interp.html>`_:
  "linear", "polynomial", "cspline" (default), "cspline_periodic", "akima",
  "akima_periodic", and "steffen". However note that some methods do not have smooth derivatives or second derivatives.

The implementation uses GSL spline routines to evaluate first and second derivatives
where appropriate. For the supported interpolation methods, GSL provides finite
derivatives:

- Linear: piecewise-constant derivative (discontinuous at knots).
- Cubic splines (cspline / cspline_periodic): continuous first and second
  derivatives.
- Akima / akima_periodic: smooth first derivative designed to reduce
  overshoots.
- Steffen: monotonic spline with continuous derivatives.
- Polynomial: smooth derivatives within the domain.

Recommendations
---------------

1. If smoothness is important, use ``cspline``. Cubic splines provide continuous
   first and second derivatives, which generally yield physically smoother densities
   when deriving them from potentials.
2. Be mindful of endpoint behavior (inherited from GSL). The cubic spline implementation
   used here attempts to make the second derivative go to zero at the boundaries (i.e.
   for the end knots). This can introduce edge effects. A simple mitigation is to place
   knots beyond the radial range where you care about the physical model (i.e., make the
   knot grid slightly larger than the region you will evaluate). This reduces the
   influence of the boundary conditions in the region of interest.
3. Use an appropriately dense knot grid in regions where the profile has high curvature
   (e.g., sharp features). Akima or steffen can be useful for reducing overshoot with
   sparse data, but these methods may produce less well-behaved higher derivatives.
4. Validate by comparing diagnostics (e.g., the enclosed mass computed from the density
   vs the supplied mass profile) and by visual checks of the recovered density/gradients
   when you change interpolation method.
5. For orbit modeling, it is generally better to supply mass profiles when possible
   (``spline_value_type='mass'``) because dPhi/dr follows directly from M(r) and is less
   sensitive to high-frequency numerical noise in derivatives.


Examples
--------

Example 1: Create and evaluate a spline potential
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a mass-based spherical spline potential and evaluate it:

.. plot::
    :align: center
    :context: close-figs

    import numpy as np
    import astropy.units as u
    import matplotlib.pyplot as plt
    from gala.units import galactic
    from gala.potential import SphericalSplinePotential

    # radial knots and enclosed mass profile (example)
    r_knots = np.logspace(-1, 2, 50) * u.kpc

    # Example mass profile (toy)
    M_r = (1e12 * u.Msun) * (r_knots / (r_knots + 10*u.kpc))**2

    pot = SphericalSplinePotential(
        r_knots=r_knots,
        spline_values=M_r,
        spline_value_type="mass",
        interpolation_method="cspline",
        units=galactic
    )

    # Evaluate at a set of radii using the r= symmetry coordinate
    r_eval = np.logspace(-1, 2, 200) * u.kpc

    phi = pot.energy(r=r_eval)
    dens = pot.density(r=r_eval)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6), layout="tight")
    ax1 = axes[0]
    ax1.semilogx(r_eval, phi)
    ax1.set_ylabel(rf"$\Phi$ [{phi.unit:latex_inline}]")

    ax2 = axes[1]
    ax2.loglog(r_eval, dens.to(u.Msun / u.kpc**3))
    ax2.set_ylabel(r"$\rho(r)$ [$M_\odot\,\mathrm{kpc}^{-3}$]")
    ax2.set_xlabel("$r$ [kpc]")


Example 2: Make a SphericalSplinePotential from a density function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example shows a more involved workflow: define a complex
analytic density profile, evaluate it on a fine radial grid, build a
``SphericalSplinePotential`` with ``spline_value_type='density'``, and plot the
resulting potential and recovered density. This is useful for quick visual
experiments and for creating documentable figures in the Sphinx docs (via the
matplotlib plot directive).

.. plot::
    :align: center
    :context: close-figs

    import numpy as np
    import astropy.units as u
    import matplotlib.pyplot as plt
    from gala.units import galactic
    from gala.potential import SphericalSplinePotential

    def rho_analytic(r):
        r = np.array(r)
        rho0 = 1e9  # Msun / kpc^3
        return (
            rho0 / r ** 1.35 / (1 + r)**3.44
        )


    # radial knots where we build the spline (note we extend beyond the region of interest)
    r_knots = (
        np.concatenate([np.logspace(-2, -0.5, 10), np.logspace(-0.5, 2.5, 100)[1:]]) * u.kpc
    )
    rho_vals = rho_analytic(r_knots.value) * u.Msun / u.kpc**3

    pot = SphericalSplinePotential(
        r_knots=r_knots,
        spline_values=rho_vals,
        spline_value_type="density",
        interpolation_method="cspline",
        units=galactic,
    )

    r_eval = np.logspace(-2, 2.3, 300) * u.kpc
    pos = (
        np.stack(
            [r_eval.value, np.zeros_like(r_eval.value), np.zeros_like(r_eval.value)], axis=0
        )
        * r_eval.unit
    )

    phi = pot.energy(pos)
    dens_recovered = pot.density(pos)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6), layout="tight")
    ax1.semilogx(r_eval, phi)
    ax1.set_ylabel(rf"$\Phi$ [{phi.unit:latex_inline}]")
    ax2.loglog(r_eval, dens_recovered.to(u.Msun / u.kpc**3), label="Recovered density")
    ax2.loglog(r_knots, rho_vals.to(u.Msun / u.kpc**3), "o", ms=3, label="Input knots")
    ax2.set_xlabel("$r$ [kpc]")
    ax2.set_ylabel(r"$\rho$ [$M_\odot\,\mathrm{kpc}^{-3}$]")
    ax2.legend()


Example 3: Effect of interpolation method (Akima vs cspline)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example constructs a potential by directly interpolating a potential-valued spline
(``spline_value_type='potential'``). We then compare the density inferred from the
potential using two different interpolation methods. Akima-style splines can produce
piecewise-smooth first derivatives that sometimes appear 'jagged' in the second
derivative (which is used to compute density), whereas cspline (cubic spline) tends to
produce continuous second derivatives.

.. plot::
    :align: center
    :context: close-figs

    import numpy as np
    import astropy.units as u
    import matplotlib.pyplot as plt
    from gala.units import galactic
    from gala.potential import SphericalSplinePotential

    # Make an example smooth potential (toy)
    r_knots = np.logspace(-1, 2, 40) * u.kpc
    phi_smooth = (
        -1e5 * (1.0 / (1.0 + (r_knots.to(u.kpc).value / 10.0) ** 2)) * u.km**2 / u.s**2
    )

    # cspline (smooth second derivative)
    pot_cspline = SphericalSplinePotential(
        r_knots=r_knots,
        spline_values=phi_smooth,
        spline_value_type="potential",
        interpolation_method="cspline",
        units=galactic,
    )

    # akima (can have less-smooth second derivative)
    pot_akima = SphericalSplinePotential(
        r_knots=r_knots,
        spline_values=phi_smooth,
        spline_value_type="potential",
        interpolation_method="akima",
        units=galactic,
    )

    r_eval = np.logspace(-1, 2, 400) * u.kpc
    pos = (
        np.stack(
            [r_eval.value, np.zeros_like(r_eval.value), np.zeros_like(r_eval.value)], axis=0
        )
        * r_eval.unit
    )

    rho_cs = pot_cspline.density(pos)
    rho_ak = pot_akima.density(pos)

    plt.figure(figsize=(6, 4))
    plt.loglog(r_eval, rho_cs.to(u.Msun / u.kpc**3), label="cspline (smooth)")
    plt.loglog(
        r_eval, rho_ak.to(u.Msun / u.kpc**3), label="akima (can appear jagged)", alpha=0.8
    )
    plt.scatter(
        r_knots, np.zeros_like(r_knots.value), marker="|", color="k", s=40, label="knots"
    )
    plt.xlabel("$r$ [kpc]")
    plt.ylabel(r"$\rho$ [$M_\odot\,\mathrm{kpc}^{-3}$]")
    plt.legend()
    plt.tight_layout()


API
---

See: `~gala.potential.potential.SphericalSplinePotential`
