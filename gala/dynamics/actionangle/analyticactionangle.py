"""
Analytic transformations to action-angle coordinates.
"""


# Third-party
import numpy as np
from astropy.constants import G
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.utils.decorators import deprecated

# Gala
import gala.dynamics as gd
from gala.util import GalaDeprecationWarning
from gala.tests.optional_deps import HAS_TWOBODY

__all__ = ['isochrone_xv_to_aa', 'isochrone_aa_to_xv',
           'harmonic_oscillator_xv_to_aa']


def F(x, y):
    z = np.zeros_like(x)

    ix = y > np.pi/2.
    z[ix] = np.pi/2. - np.arctan(np.tan(np.pi/2. - 0.5*y[ix])/x[ix])

    ix = y < -np.pi/2.
    z[ix] = -np.pi/2. + np.arctan(np.tan(np.pi/2. + 0.5*y[ix])/x[ix])

    ix = (y <= np.pi/2) & (y >= -np.pi/2)
    z[ix] = np.arctan(x[ix] * np.tan(0.5*y[ix]))
    return z


def isochrone_xv_to_aa(w, potential):
    """
    Transform the input cartesian position and velocity to action-angle
    coordinates in the Isochrone potential. See Section 3.5.2 in
    Binney & Tremaine (2008), and be aware of the errata entry for
    Eq. 3.225.

    This transformation is analytic and can be used as a "toy potential"
    in the Sanders & Binney (2014) formalism for computing action-angle
    coordinates in any potential.

    Parameters
    ----------
    w : :class:`gala.dynamics.PhaseSpacePosition`, :class:`gala.dynamics.Orbit`
    potential : :class:`gala.potential.IsochronePotential`, dict
        An instance of the potential to use for computing the transformation
        to angle-action coordinates. Or, a dictionary of parameters used to
        define an :class:`gala.potential.IsochronePotential` instance.

    Returns
    -------
    actions : :class:`~astropy.units.Quantity`
        Actions computed from the input positions and velocities.
    angles : :class:`~astropy.units.Quantity`
        Angles computed from the input positions and velocities.
    freqs : :class:`~astropy.units.Quantity`
        Frequencies computed from the input positions and velocities.
    """
    from gala.potential import Hamiltonian, PotentialBase, IsochronePotential

    if not isinstance(potential, PotentialBase):
        potential = IsochronePotential(**potential)

    usys = potential.units
    GM = (G*potential.parameters['m']).decompose(usys).value
    b = potential.parameters['b'].decompose(usys).value
    E = w.energy(Hamiltonian(potential)).decompose(usys).value
    E = np.atleast_1d(E)

    if np.any(E > 0.):
        raise ValueError("Unbound particle. (E = {})".format(E))

    # convert position, velocity to spherical polar coordinates
    w_sph = w.represent_as(coord.PhysicsSphericalRepresentation)
    r, phi, theta = map(
        np.atleast_1d,
        [
            w_sph.r.decompose(usys).value,
            w_sph.phi.radian,
            w_sph.theta.radian
        ]
    )

    ang_unit = u.radian/usys['time']
    vr, phi_dot, theta_dot = map(
        np.atleast_1d,
        [
            w_sph.radial_velocity.decompose(usys).value,
            w_sph.pm_phi.to(ang_unit).value,
            w_sph.pm_theta.to(ang_unit).value
        ]
    )
    vtheta = r * theta_dot

    # ----------------------------
    # Compute the actions
    # ----------------------------

    L_vec = [np.atleast_1d(x)
             for x in w.angular_momentum().decompose(usys).value]
    Lz = L_vec[2]
    L = np.linalg.norm(L_vec, axis=0)

    # Radial action
    Jr = GM / np.sqrt(-2*E) - 0.5*(L + np.sqrt(L*L + 4*GM*b))

    # compute the three action variables
    actions = np.array([Jr, Lz, L - np.abs(Lz)]).reshape((3,) + w.shape)

    # ----------------------------
    # Angles
    # ----------------------------
    c = GM / (-2*E) - b
    e = np.sqrt(1 - L*L*(1 + b/c) / GM / c)

    # Compute theta_r using eta
    tmp1 = r*vr / np.sqrt(-2.*E)
    tmp2 = b + c - np.sqrt(b*b + r*r)
    eta = np.arctan2(tmp1, tmp2)
    thetar = eta - e*c*np.sin(eta) / (c + b)  # same as theta3

    # Compute theta_z
    psi = np.arctan2(np.cos(theta), -np.sin(theta)*r*vtheta/L)
    psi[np.abs(vtheta) <= 1e-10] = np.pi/2.  # blows up for small vtheta

    omega_ratio = 0.5 * (1 + L/np.sqrt(L*L + 4*GM*b))

    a = np.sqrt((1+e) / (1-e))
    ap = np.sqrt((1 + e + 2*b/c) / (1 - e + 2*b/c))

    A = omega_ratio*thetar - F(a, eta) - F(ap, eta)/np.sqrt(1 + 4*GM*b/L/L)
    thetat = psi + A

    LR = Lz/L
    sinu = (LR/np.sqrt(1.-LR*LR)/np.tan(theta))
    uu = np.arcsin(sinu)

    uu[sinu > 1.] = np.pi/2.
    uu[sinu < -1.] = -np.pi/2.
    uu[vtheta > 0.] = np.pi - uu[vtheta > 0.]

    thetap = phi - uu + np.sign(Lz)*thetat
    angles = np.array([thetar, thetap, thetat]).reshape((3,) + w.shape)
    angles = angles % (2*np.pi)

    # ----------------------------
    # Frequencies
    # ----------------------------
    freqs = np.zeros_like(actions)
    omega_r = GM**2 / (Jr + 0.5*(L + np.sqrt(L*L + 4*GM*b)))**3
    freqs[0] = omega_r
    freqs[1] = np.sign(actions[1]) * omega_ratio * omega_r
    freqs[2] = omega_ratio * omega_r

    a_unit = (1 * usys['angular momentum'] / usys['mass']).decompose(usys).unit
    f_unit = (1 * usys['angular speed']).decompose(usys).unit
    return actions * a_unit, angles * u.radian, freqs * f_unit


@deprecated(since="v1.5",
            name="isochrone_to_aa",
            alternative="isochrone_xv_to_aa",
            warning_type=GalaDeprecationWarning)
def isochrone_to_aa(*args, **kwargs):
    """
    Deprecated! Use `gala.dynamics.actionangle.isochrone_xv_to_aa` instead.
    """
    return isochrone_xv_to_aa(*args, **kwargs)


def isochrone_aa_to_xv(actions, angles, potential):
    """
    Transform the input actions and angles to cartesian position and velocity
    coordinates in the Isochrone potential. See Section 3.5.2 in
    Binney & Tremaine (2008), and be aware of the errata entry for
    Eq. 3.225.

    Parameters
    ----------
    actions : :class:`~astropy.units.Quantity`
    angles : :class:`~astropy.units.Quantity`
    potential : :class:`gala.potential.IsochronePotential`, dict
        An instance of the potential to use for computing the transformation
        to angle-action coordinates. Or, a dictionary of parameters used to
        define an :class:`gala.potential.IsochronePotential` instance.

    Returns
    -------
    w : :class:`gala.dynamics.PhaseSpacePosition`
        The computed positions and velocities.
    """
    if not HAS_TWOBODY:
        raise ImportError(
            "Failed to import twobody: Converting from action-angle "
            "coordinates to position and velocity in the isochrone potential "
            "requires a Kepler solver, and thus `twobody` must be installed.")

    import twobody as tb

    Jr, Jphi, Jth = [np.atleast_1d(x) for x in actions]
    thr, thphi, thth = [np.atleast_1d(x) for x in angles]

    GM = (G * potential.parameters['m'])
    b = potential.parameters['b']

    Lz = Jphi
    L = Jth + np.abs(Lz)

    # Eq.3.225 in B&T 2008
    sqrt_L2_4GMb = np.sqrt(L**2 + 4 * GM * b)
    E = -0.5 * (GM / (Jr + 0.5 * (L + sqrt_L2_4GMb)))**2

    # Coordinates orientation crap
    i = np.arccos(Lz / L)
    lon_nodes = coord.Angle(thphi - np.sign(Lz) * thth).wrap_at(2*np.pi*u.rad)
    # TODO: could check that std(i), std(lon_nodes) are small...

    # Auxiliary variables (Eq. 3.240)
    c = GM / (-2*E) - b
    e = np.sqrt(1 - L**2 / (GM * c) * (1 + b/c))

    e_eff = e * c / (c + b)
    eta = tb.eccentric_anomaly_from_mean_anomaly(thr, e_eff)

    s = 2 + c/b * (1 - e * np.cos(eta))
    r = b * np.sqrt((s - 1)**2 - 1)

    Omr = GM**2 / (Jr + 0.5 * (L + sqrt_L2_4GMb))**3
    eta_dot = Omr / (1 - e_eff * np.cos(eta))
    s_dot = e * c / b * np.sin(eta) * eta_dot
    vr = b * (s - 1) * s_dot / np.sqrt((s-1)**2 - 1)
    v_tan = L / r

    sqrt1 = np.sqrt(1 + e) / np.sqrt(1-e)
    sqrt2 = np.sqrt(1 + e + 2*b/c) / np.sqrt(1 - e + 2*b/c)

    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        terms = (
            0.5 * (1 + L / sqrt_L2_4GMb) * thr
            - F(sqrt1, eta)
            - L / sqrt_L2_4GMb * F(sqrt2, eta)
        )
    # psi = angles[2] - terms
    psi = thth - terms - 3*np.pi/2*u.rad  # WT actual F

    xyz_prime = (np.array([
        r.value * np.cos(psi),
        r.value * np.sin(psi),
        np.zeros_like(r.value)]
    ) * r.unit).to(potential.units['length'])

    vx = vr * np.cos(psi) - v_tan * np.sin(psi)
    vy = vr * np.sin(psi) + v_tan * np.cos(psi)
    vxyz_prime = (np.array([
        vx.value,
        vy.to_value(vx.unit),
        np.zeros_like(r.value)
    ]) * vx.unit).to(potential.units['velocity'])

    M1 = rotation_matrix(-i, 'y')
    M2 = rotation_matrix(-lon_nodes, 'z')
    M3 = rotation_matrix(np.pi/2 * u.rad, 'z')  # WT actual F
    M = np.einsum('ij,...jk,...kl->...il', M3, M2, M1)

    xyz = np.einsum('...ij,j...->i...', M, xyz_prime)
    vxyz = np.einsum('...ij,j...->i...', M, vxyz_prime)

    w = gd.PhaseSpacePosition(pos=xyz, vel=vxyz)

    return w.reshape(actions.shape[1:])


def harmonic_oscillator_xv_to_aa(w, potential):
    """
    Transform the input cartesian position and velocity to action-angle
    coordinates for the Harmonic Oscillator potential.

    This transformation is analytic and can be used as a "toy potential"
    in the Sanders & Binney (2014) formalism for computing action-angle
    coordinates in any potential.

    Parameters
    ----------
    w : :class:`gala.dynamics.PhaseSpacePosition`, :class:`gala.dynamics.Orbit`
    potential : Potential

    Returns
    -------
    actions : :class:`~astropy.units.Quantity`
        Actions computed from the input positions and velocities.
    angles : :class:`~astropy.units.Quantity`
        Angles computed from the input positions and velocities.
    freqs : :class:`~astropy.units.Quantity`
        Frequencies computed from the input positions and velocities.
    """

    usys = potential.units
    if usys is not None:
        x = w.xyz.decompose(usys).value
        v = w.v_xyz.decompose(usys).value
    else:
        x = w.xyz.value
        v = w.v_xyz.value
    _new_omega_shape = (3,) + tuple([1]*(len(x.shape)-1))

    # compute actions -- just energy (hamiltonian) over frequency
    if usys is None:
        usys = []

    try:
        omega = potential.parameters['omega'].reshape(_new_omega_shape).decompose(usys).value
    except AttributeError:  # not a Quantity
        omega = potential.parameters['omega'].reshape(_new_omega_shape)

    action = (v**2 + (omega*x)**2) / (2.*omega)

    angle = np.arctan(-v / omega / x)
    angle[x == 0] = -np.sign(v[x == 0])*np.pi/2.
    angle[x < 0] += np.pi

    freq = potential.parameters['omega'].decompose(usys).value

    if usys is not None and usys:
        a_unit = (1*usys['angular momentum']/usys['mass']).decompose(usys).unit
        f_unit = (1*usys['angular speed']).decompose(usys).unit
        return action * a_unit, (angle % (2.*np.pi)) * u.radian, freq * f_unit
    else:
        return action * u.one, (angle % (2.*np.pi)) * u.one, freq * u.one


@deprecated(since="v1.5",
            name="harmonic_oscillator_to_aa",
            alternative="harmonic_oscillator_xv_to_aa",
            warning_type=GalaDeprecationWarning)
def harmonic_oscillator_to_aa(*args, **kwargs):
    """
    Deprecated! Use `gala.dynamics.actionangle.harmonic_oscillator_xv_to_aa`
    instead.
    """
    return harmonic_oscillator_xv_to_aa(*args, **kwargs)


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
    # omega = potential.parameters['omega'].decompose(potential.units).value
    # x = np.sqrt(2*actions/omega[None]) * np.sin(angles)
    # v = np.sqrt(2*actions*omega[None]) * np.cos(angles)

    # return x, v
