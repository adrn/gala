# coding: utf-8

from __future__ import division, print_function

"""
Analytic transformations to action-angle coordinates.
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy.constants import G

# Project
from ..coordinates import cartesian_to_spherical, spherical_to_cartesian
from .core import angular_momentum

__all__ = ['isochrone_xv_to_aa', 'isochrone_aa_to_xv',
           'harmonic_oscillator_xv_to_aa', 'harmonic_oscillator_aa_to_xv']

def isochrone_xv_to_aa(x, v, potential):
    """
    Transform the input cartesian position and velocity to action-angle
    coordinates in the Isochrone potential. See Section 3.5.2 in
    Binney & Tremaine (2008), and be aware of the errata entry for
    Eq. 3.225.

    This transformation is analytic and can be used as a "toy potential"
    in the Sanders & Binney (2014) formalism for computing action-angle
    coordinates in any potential.

    .. note::

        This function is included as a method of the :class:`~gary.potential.IsochronePotential`
        and it is recommended to call :meth:`~gary.potential.IsochronePotential.phase_space()`
        instead.

    Parameters
    ----------
    x : array_like
        Cartesian positions. Must have shape (N,3) or (3,).
    v : array_like
        Cartesian velocities. Must have shape (N,3) or (3,).
    potential : :class:`gary.potential.IsochronePotential`
        An instance of the potential to use for computing the transformation
        to angle-action coordinates.

    Returns
    -------
    actions : :class:`numpy.ndarray`
        An array of actions computed from the input positions and velocities. Will
        always have shape (N,3) -- if input coordinates are 1D, the output shape will
        be (1,3).
    angles : :class:`numpy.ndarray`
        An array of angles computed from the input positions and velocities. Will
        always have shape (N,3) -- if input coordinates are 1D, the output shape will
        be (1,3).
    """

    x = np.atleast_2d(x)
    v = np.atleast_2d(v)

    _G = G.decompose(potential.units).value
    GM = _G*potential.parameters['m']
    b = potential.parameters['b']
    E = potential.total_energy(x, v)

    if np.any(E > 0.):
        raise ValueError("Unbound particle. (E = {})".format(E))

    # convert position, velocity to spherical polar coordinates
    sph = cartesian_to_spherical(x, v)
    r,phi,theta,vr,vphi,vtheta = sph.T

    # ----------------------------
    # Actions
    # ----------------------------

    L_vec = angular_momentum(np.hstack((x,v)))
    Lz = L_vec[:,2]
    L = np.linalg.norm(L_vec, axis=1)

    # Radial action
    Jr = GM / np.sqrt(-2*E) - 0.5*(L + np.sqrt(L*L + 4*GM*b))

    # compute the three action variables
    actions = np.array([Jr, Lz, L - np.abs(Lz)]).T

    # ----------------------------
    # Angles
    # ----------------------------
    c = GM / (-2*E) - b
    e = np.sqrt(1 - L*L*(1 + b/c) / GM / c)

    # Compute theta_r using eta
    tmp1 = r*vr / np.sqrt(-2.*E)
    tmp2 = b + c - np.sqrt(b*b + r*r)
    eta = np.arctan2(tmp1,tmp2)
    thetar = eta - e*c*np.sin(eta) / (c + b)  # same as theta3

    # Compute theta_z
    psi = np.arctan2(np.cos(theta), -np.sin(theta)*r*vtheta/L)
    psi[np.abs(vtheta) <= 1e-10] = np.pi/2.  # blows up for small vtheta

    omega = 0.5 * (1 + L/np.sqrt(L*L + 4*GM*b))

    a = np.sqrt((1+e) / (1-e))
    ap = np.sqrt((1 + e + 2*b/c) / (1 - e + 2*b/c))

    def F(x, y):
        z = np.zeros_like(x)

        ix = y>np.pi/2.
        z[ix] = np.pi/2. - np.arctan(np.tan(np.pi/2.-0.5*y[ix])/x[ix])

        ix = y<-np.pi/2.
        z[ix] = -np.pi/2. + np.arctan(np.tan(np.pi/2.+0.5*y[ix])/x[ix])

        ix = (y<=np.pi/2) & (y>=-np.pi/2)
        z[ix] = np.arctan(x[ix]*np.tan(0.5*y[ix]))
        return z

    A = omega*thetar - F(a,eta) - F(ap,eta)/np.sqrt(1 + 4*GM*b/L/L)
    thetaz = psi + A

    LR = Lz/L
    sinu = LR/np.sqrt(1.-LR*LR)/np.tan(theta)

    u = np.arcsin(sinu)
    u[sinu > 1.] = np.pi/2.
    u[sinu < -1.] = -np.pi/2.
    u[vtheta > 0.] = np.pi - u[vtheta > 0.]

    thetap = phi - u + np.sign(Lz)*thetaz
    angles = np.array([thetar, thetap, thetaz]).T
    angles %= (2*np.pi)

    return actions, angles

def isochrone_aa_to_xv(actions, angles, potential):
    """
    Transform the input actions and angles to ordinary phase space (position
    and velocity) in cartesian coordinates. See Section 3.5.2 in
    Binney & Tremaine (2008), and be aware of the errata entry for
    Eq. 3.225.

    .. note::

        This function is included as a method of the :class:`~gary.potential.IsochronePotential`
        and it is recommended to call :meth:`~gary.potential.IsochronePotential.action_angle()`
        instead.

    Parameters
    ----------
    actions : array_like
        Action variables. Must have shape (N,3) or (3,).
    angles : array_like
        Angle variables. Must have shape (N,3) or (3,).
    potential : :class:`gary.potential.IsochronePotential`
        An instance of the potential to use for computing the transformation
        to angle-action coordinates.

    Returns
    -------
    x : :class:`numpy.ndarray`
        An array of cartesian positions computed from the input angles and actions. Will
        always have shape (N,3) -- if input coordinates are 1D, the output shape will
        be (1,3).
    v : :class:`numpy.ndarray`
        An array of cartesian velocities computed from the input angles and actions. Will
        always have shape (N,3) -- if input coordinates are 1D, the output shape will
        be (1,3).

    """

    actions = np.atleast_2d(actions)
    angles = np.atleast_2d(angles)

    _G = G.decompose(potential.units).value
    GM = _G*potential.parameters['m']
    b = potential.parameters['b']

    # actions
    Jr = actions[:,0]
    Lz = actions[:,1]
    L = actions[:,2] + np.abs(Lz)

    # angles
    theta_r,theta_phi,theta_theta = angles.T

    # get longitude of ascending node
    theta_1 = theta_phi - np.sign(Lz)*theta_theta
    Omega = theta_1

    # Ly = -np.cos(Omega) * np.sqrt(L**2 - Lz**2)
    # Lx = np.sqrt(L**2 - Ly**2 - Lz**2)
    cosi = Lz/L
    sini = np.sqrt(1 - cosi**2)

    # Hamiltonian (energy)
    H = -2. * GM**2 / (2.*Jr + L + np.sqrt(4.*b*GM + L**2))**2

    if np.any(H > 0.):
        raise ValueError("Unbound particle. (E = {})".format(H))

    # Eq. 3.240
    c = -GM / (2.*H) - b
    e = np.sqrt(1 - L*L*(1 + b/c) / GM / c)

    # solve for eta
    theta_3 = theta_r
    eta_func = lambda x: x - e*c/(b+c)*np.sin(x) - theta_3
    eta_func_prime = lambda x: 1 - e*c/(b+c)*np.cos(x)

    # use newton's method to find roots
    niter = 100
    eta = np.ones_like(theta_3)*np.pi/2.
    for i in range(niter):
        eta -= eta_func(eta)/eta_func_prime(eta)

    # TODO: when to do this???
    eta -= 2*np.pi

    r = c*np.sqrt((1-e*np.cos(eta)) * (1-e*np.cos(eta) + 2*b/c))
    vr = np.sqrt(GM/(b+c))*(c*e*np.sin(eta))/r

    theta_2 = theta_theta
    Omega_23 = 0.5*(1 + L / np.sqrt(L**2 + 4*GM*b))

    a = np.sqrt((1+e) / (1-e))
    ap = np.sqrt((1 + e + 2*b/c) / (1 - e + 2*b/c))

    def F(x, y):
        z = np.zeros_like(x)

        ix = y>np.pi/2.
        z[ix] = np.pi/2. - np.arctan(np.tan(np.pi/2.-0.5*y[ix])/x[ix])

        ix = y<-np.pi/2.
        z[ix] = -np.pi/2. + np.arctan(np.tan(np.pi/2.+0.5*y[ix])/x[ix])

        ix = (y<=np.pi/2) & (y>=-np.pi/2)
        z[ix] = np.arctan(x[ix]*np.tan(0.5*y[ix]))
        return z

    theta_2[Lz < 0] -= 2*np.pi
    theta_3 -= 2*np.pi
    A = Omega_23*theta_3 - F(a,eta) - F(ap,eta)/np.sqrt(1 + 4*GM*b/L/L)
    psi = theta_2 - A

    # theta
    theta = np.arccos(np.sin(psi)*sini)
    vtheta = L*sini*np.cos(psi)/np.cos(theta)
    vtheta = -L*sini*np.cos(psi)/np.sin(theta)/r
    vphi = Lz / (r*np.sin(theta))

    # phi
    sinu = np.sin(psi)*cosi/np.sin(theta)
    u = np.arcsin(sinu)
    u[sinu > 1.] = np.pi/2.
    u[sinu < -1.] = -np.pi/2.
    u[vtheta > 0.] = np.pi - u[vtheta > 0.]

    sinu = cosi/sini * np.cos(theta)/np.sin(theta)
    phi = (u + Omega) % (2*np.pi)

    # We now need to convert from spherical polar coord to cart. coord.
    x,v = spherical_to_cartesian(r,phi,theta,vr,vphi,vtheta)
    return x,v

def harmonic_oscillator_xv_to_aa(x, v, potential):
    """
    Transform the input cartesian position and velocity to action-angle
    coordinates the Harmonic Oscillator potential. This transformation
    is analytic and can be used as a "toy potential" in the
    Sanders & Binney 2014 formalism for computing action-angle coordinates
    in _any_ potential.

    Adapted from Jason Sanders' code
    `genfunc <https://github.com/jlsanders/genfunc>`_.

    Parameters
    ----------
    x : array_like
        Positions.
    v : array_like
        Velocities.
    potential : Potential
    """
    omega = np.atleast_2d(potential.parameters['omega'])

    # compute actions -- just energy (hamiltonian) over frequency
    # E = potential.total_energy(x,v)[:,None]
    omega = potential.parameters['omega']
    action = (v**2 + (omega*x)**2)/(2.*omega[None])

    angle = np.arctan(-v / omega / x)
    angle[x == 0] = -np.sign(v[x == 0])*np.pi/2.
    angle[x < 0] += np.pi

    return action, angle % (2.*np.pi)

def harmonic_oscillator_aa_to_xv(actions, angles, potential):
    """
    Transform the input action-angle coordinates to cartesian position and velocity
    assuming a Harmonic Oscillator potential. This transformation
    is analytic and can be used as a "toy potential" in the
    Sanders & Binney 2014 formalism for computing action-angle coordinates
    in _any_ potential.

    Adapted from Jason Sanders' code
    `genfunc <https://github.com/jlsanders/genfunc>`_.

    Parameters
    ----------
    actions : array_like
    angles : array_like
    potential : Potential
    """
    raise NotImplementedError()

    # TODO: bug in below...
    omega = potential.parameters['omega']
    x = np.sqrt(2*actions/omega[None]) * np.sin(angles)
    v = np.sqrt(2*actions*omega[None]) * np.cos(angles)

    return x,v
