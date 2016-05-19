# coding: utf-8

from __future__ import division, print_function

"""
Analytic transformations to action-angle coordinates.
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy.constants import G
import astropy.coordinates as coord
import astropy.units as u

# Project
from ..potential import PotentialBase, IsochronePotential, HarmonicOscillatorPotential
from ..coordinates import physicsspherical_to_cartesian
from ..util import atleast_2d

__all__ = ['isochrone_to_aa', 'harmonic_oscillator_to_aa']

def isochrone_to_aa(w, potential):
    """
    Transform the input cartesian position and velocity to action-angle
    coordinates in the Isochrone potential. See Section 3.5.2 in
    Binney & Tremaine (2008), and be aware of the errata entry for
    Eq. 3.225.

    This transformation is analytic and can be used as a "toy potential"
    in the Sanders & Binney (2014) formalism for computing action-angle
    coordinates in any potential.

    .. note::

        This function is included as a method of the
        :class:`~gala.potential.IsochronePotential` and it is recommended
        to call :meth:`~gala.potential.IsochronePotential.phase_space()`
        instead.

    Parameters
    ----------
    w : :class:`gala.dynamics.CartesianPhaseSpacePosition`, :class:`gala.dynamics.CartesianOrbit`
    potential : :class:`gala.potential.IsochronePotential`, dict
        An instance of the potential to use for computing the transformation
        to angle-action coordinates. Or, a dictionary of parameters used to
        define an :class:`gala.potential.IsochronePotential` instance.

    Returns
    -------
    actions : :class:`numpy.ndarray`
        An array of actions computed from the input positions and velocities.
    angles : :class:`numpy.ndarray`
        An array of angles computed from the input positions and velocities.
    freqs : :class:`numpy.ndarray`
        An array of frequencies computed from the input positions and velocities.
    """

    if not isinstance(potential, PotentialBase):
        potential = IsochronePotential(**potential)

    usys = potential.units
    GM = (G*potential.parameters['m']).decompose(usys).value
    b = potential.parameters['b'].decompose(usys).value
    E = w.energy(potential).decompose(usys).value

    if np.any(E > 0.):
        raise ValueError("Unbound particle. (E = {})".format(E))

    # convert position, velocity to spherical polar coordinates
    sph,vsph = w.represent_as(coord.PhysicsSphericalRepresentation)
    r,phi,theta = sph.r.value, sph.phi.value, sph.theta.value
    vr,vphi,vtheta = vsph.value

    # ----------------------------
    # Compute the actions
    # ----------------------------

    L_vec = w.angular_momentum().decompose(usys).value
    Lz = L_vec[2]
    L = np.linalg.norm(L_vec, axis=0)

    # Radial action
    Jr = GM / np.sqrt(-2*E) - 0.5*(L + np.sqrt(L*L + 4*GM*b))

    # compute the three action variables
    actions = np.array([Jr, Lz, L - np.abs(Lz)]) # Jr, Jphi, Jtheta

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

    omega_th = 0.5 * (1 + L/np.sqrt(L*L + 4*GM*b))

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

    A = omega_th*thetar - F(a,eta) - F(ap,eta)/np.sqrt(1 + 4*GM*b/L/L)
    thetaz = psi + A

    LR = Lz/L
    sinu = (LR/np.sqrt(1.-LR*LR)/np.tan(theta))
    sinu = sinu
    uu = np.arcsin(sinu)

    uu[sinu > 1.] = np.pi/2.
    uu[sinu < -1.] = -np.pi/2.
    uu[vtheta > 0.] = np.pi - uu[vtheta > 0.]

    thetap = phi - uu + np.sign(Lz)*thetaz
    angles = np.array([thetar, thetap, thetaz])
    angles = angles % (2*np.pi)

    # ----------------------------
    # Frequencies
    # ----------------------------
    freqs = np.zeros_like(actions)
    omega_r = GM**2 / (Jr + 0.5*(L + np.sqrt(L*L + 4*GM*b)))**3
    freqs[0] = omega_r
    freqs[1] = np.sign(actions[1]) * omega_th * omega_r
    freqs[2] = omega_th * omega_r

    a_unit = (1*usys['angular momentum']).decompose(usys).unit
    f_unit = (1*usys['frequency']).decompose(usys).unit
    return actions*a_unit, angles*u.radian, freqs*f_unit

def isochrone_to_xv(actions, angles, potential):
    """
    Transform the input actions and angles to ordinary phase space (position
    and velocity) in cartesian coordinates. See Section 3.5.2 in
    Binney & Tremaine (2008), and be aware of the errata entry for
    Eq. 3.225.

    .. note::

        This function is included as a method of the :class:`~gala.potential.IsochronePotential`
        and it is recommended to call :meth:`~gala.potential.IsochronePotential.action_angle()`
        instead.

    Parameters
    ----------
    actions : array_like
        Action variables. Must have shape ``(3,N)`` or ``(3,)``.
    angles : array_like
        Angle variables. Must have shape ``(3,N)`` or ``(3,)``.
        Should be in radians.
    potential : :class:`gala.potential.IsochronePotential`
        An instance of the potential to use for computing the transformation
        to angle-action coordinates.

    Returns
    -------
    x : :class:`numpy.ndarray`
        An array of cartesian positions computed from the input
        angles and actions.
    v : :class:`numpy.ndarray`
        An array of cartesian velocities computed from the input
        angles and actions.
    """

    raise NotImplementedError("Implementation not supported until working with "
                              "angle-action variables has a better API.")

    actions = atleast_2d(actions,insert_axis=1).copy()
    angles = atleast_2d(angles,insert_axis=1).copy()

    usys = potential.units
    GM = (G*potential.parameters['m']).decompose(usys).value
    b = potential.parameters['b'].decompose(usys).value

    # actions
    Jr = actions[0]
    Lz = actions[1]
    L = actions[2] + np.abs(Lz)

    # angles
    theta_r,theta_phi,theta_theta = angles

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

    uu = np.arcsin(sinu)
    uu[sinu > 1.] = np.pi/2.
    uu[sinu < -1.] = -np.pi/2.
    uu[vtheta > 0.] = np.pi - uu[vtheta > 0.]

    sinu = cosi/sini * np.cos(theta)/np.sin(theta)
    phi = (uu + Omega) % (2*np.pi)

    # We now need to convert from spherical polar coord to cart. coord.
    pos = coord.PhysicsSphericalRepresentation(r=r*u.dimensionless_unscaled,
                                               phi=phi*u.rad, theta=theta*u.rad)
    x = pos.represent_as(coord.CartesianRepresentation).xyz.value
    v = physicsspherical_to_cartesian(pos, [vr,vphi,vtheta]*u.dimensionless_unscaled).value
    return x,v

def harmonic_oscillator_to_aa(w, potential):
    """
    Transform the input cartesian position and velocity to action-angle
    coordinates for the Harmonic Oscillator potential.

    This transformation is analytic and can be used as a "toy potential"
    in the Sanders & Binney (2014) formalism for computing action-angle
    coordinates in any potential.

    .. note::

        This function is included as a method of the
        :class:`~gala.potential.HarmonicOscillatorPotential`
        and it is recommended to call
        :meth:`~gala.potential.HarmonicOscillatorPotential.action_angle()` instead.

    Parameters
    ----------
    w : :class:`gala.dynamics.CartesianPhaseSpacePosition`, :class:`gala.dynamics.CartesianOrbit`
    potential : Potential
    """

    usys = potential.units
    if usys is not None:
        x = w.pos.decompose(usys).value
        v = w.vel.decompose(usys).value
    else:
        x = w.pos.value
        v = w.vel.value
    _new_omega_shape = (3,) + tuple([1]*(len(x.shape)-1))

    # compute actions -- just energy (hamiltonian) over frequency
    if usys is None:
        usys = []

    try:
        omega = potential.parameters['omega'].reshape(_new_omega_shape).decompose(usys).value
    except AttributeError: # not a Quantity
        omega = potential.parameters['omega'].reshape(_new_omega_shape)

    action = (v**2 + (omega*x)**2)/(2.*omega)

    angle = np.arctan(-v / omega / x)
    angle[x == 0] = -np.sign(v[x == 0])*np.pi/2.
    angle[x < 0] += np.pi

    freq = potential.parameters['omega'].decompose(usys).value

    if usys is not None and usys:
        a_unit = (1*usys['angular momentum']).decompose(usys).unit
        f_unit = (1*usys['frequency']).decompose(usys).unit
        return action*a_unit, (angle % (2.*np.pi))*u.radian, freq*f_unit
    else:
        return action*u.one, (angle % (2.*np.pi))*u.one, freq*u.one

def harmonic_oscillator_to_xv(actions, angles, potential):
    """
    Transform the input action-angle coordinates to cartesian
    position and velocity for the Harmonic Oscillator potential.

    .. note::

        This function is included as a method of the
        :class:`~gala.potential.HarmonicOscillatorPotential`
        and it is recommended to call
        :meth:`~gala.potential.HarmonicOscillatorPotential.phase_space()` instead.

    Parameters
    ----------
    actions : array_like
    angles : array_like
    potential : Potential
    """
    raise NotImplementedError("Implementation not supported until working with "
                              "angle-action variables has a better API.")

    # TODO: bug in below...
    omega = potential.parameters['omega'].decompose(potential.units).value
    x = np.sqrt(2*actions/omega[None]) * np.sin(angles)
    v = np.sqrt(2*actions*omega[None]) * np.cos(angles)

    return x,v
