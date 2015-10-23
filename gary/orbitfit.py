# coding: utf-8

""" Utilities for fitting orbits to stream data. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.coordinates as coord
from astropy.coordinates.angles import rotation_matrix
import astropy.units as u
uno = u.dimensionless_unscaled
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import InterpolatedUnivariateSpline

# Project
from .coordinates import Quaternion, vgal_to_hel
from .units import galactic

def _rotation_opt_func(qua_wxyz, xyz):
    """
    Given a quaternion vector ``(w,x,y,z)`` and the data in heliocentric,
    Cartesian coordinates, compute the cost function for the given rotation.
    """
    R = Quaternion(qua_wxyz).rotation_matrix
    sph = coord.CartesianRepresentation(R.dot(xyz)*uno)\
               .represent_as(coord.SphericalRepresentation)
    return np.sum(sph.lat.degree**2)

def compute_stream_rotation_matrix(coordinate, wxyz0=None, align_lon=False):
    """
    Compute the rotation matrix to go from the frame of the input
    coordinate to closely align the equator with the stream.

    Parameter
    ---------
    coordinate : :class:`astropy.coordinate.SkyCoord`, :class:`astropy.coordinate.BaseCoordinateFrame`
        The coordinates of the stream stars.
    wxyz0 : array_like (optional)
        Initial guess for the quaternion vector that represents the rotation.
    align_lon : bool (optional)
        Also rotate in longitude so the minimum longitude is 0.

    Returns
    -------
    R : :class:`~numpy.ndarray`
        A 3 by 3 rotation matrix (has shape ``(3,3)``) to convert heliocentric,
        Cartesian coordinates in the input coordinate frame to stream coordinates.
    """
    if wxyz0 is None:
        wxyz0 = Quaternion.random().wxyz

    res = minimize(_rotation_opt_func, x0=wxyz0, args=(coordinate.cartesian.xyz,))
    R = Quaternion(res.x).rotation_matrix

    if align_lon:
        new_xyz = R.dot(coordinate.cartesian.xyz.value)
        lon = np.arctan2(new_xyz[1], new_xyz[0])
        R2 = rotation_matrix(lon.min()*u.radian, 'z')
        R = R2*R

    return R

def rotate_sph_coordinate(rep, R):
    """

    Parameter
    ---------
    rep : :class:`astropy.coordinate.BaseCoordinateFrame`, :class:`astropy.coordinate.BaseRepresentation`
        The coordinates.
    R : array_like
        The rotation matrix.

    Returns
    -------
    sph : :class:`astropy.coordinates.SphericalRepresentation`

    """
    if hasattr(rep, 'xyz'):
        xyz = rep.xyz.value
        unit = rep.xyz.unit
    elif hasattr(rep, 'represent_as'):
        xyz = rep.represent_as(coord.CartesianRepresentation).xyz
        unit = xyz.unit
        xyz = xyz.value
    else:
        raise TypeError("Input rep must be either a BaseRepresentation or a "
                        "BaseCoordinateFrame subclass.")

    sph = coord.CartesianRepresentation(R.dot(xyz)*unit)\
               .represent_as(coord.SphericalRepresentation)
    return sph

# ----------------------------------------------------------------------------
# For inference:

def ln_prior(p):
    """
    Evaluate the prior over stream orbit fit parameters.

    Parameters
    ----------
    p : iterable
        The parameters of the model: the 6 orbital initial conditions, the integration time,
        intrinsic angular width of the stream.
    """
    w0 = p[:6]
    t_integ = p[6]
    phi2_sigma = p[7] # intrinsic width on sky

    lp = 0.

    # prior on instrinsic width of stream
    if phi2_sigma <= 0.:
        return -np.inf
    lp += -np.log(phi2_sigma)

    # prefer shorter integrations
    if t_integ <= 0.1 or t_integ > 1000.: # 1 Myr to 1000 Myr
        return -np.inf
    lp += -np.log(t_integ)

    return lp

def ln_likelihood(p, data_coord, data_veloc, data_uncer, potential, dt, R, reference_frame=dict()):
    """
    Evaluate the stream orbit fit likelihood.

    Parameters
    ----------
    p : iterable
        The parameters of the model: the 6 orbital initial conditions, the integration time,
        intrinsic angular width of the stream.
    data_coord : :class:`astropy.coordinate.SkyCoord`, :class:`astropy.coordinate.BaseCoordinateFrame`
    data_veloc : iterable
        An iterable of Astropy :class:`astropy.units.Quantity` objects for the proper motions and
        line-of-sight velocity. The proper motions should be in the same coordinate frame as
        ``data_coord``.
    data_uncer : iterable
        An iterable of Astropy :class:`astropy.units.Quantity` objects for the uncertainties in
        each observable. Should have length = 6.
    potential : :class:`gary.potential.PotentialBase`
        The gravitational potential.
    dt : float
        Timestep for integrating the orbit.
    R : :class:`numpy.ndarray`
        The rotation matrix to convert from the coordinate frame of ``data_coord`` to stream
        coordinates.
    reference_frame : dict (optional)
        Any parameters that specify the reference frame, such as the Sun-Galactic Center distance,
        the circular velocity of the Sun, etc.

    Returns
    -------
    ll : :class:`numpy.ndarray`
        An array of likelihoods for each data point.

    """
    w0 = p[:6]
    t_integ = p[6]
    phi2_sigma = p[7] # intrinsic width on sky

    # integrate orbit
    t,w = potential.integrate_orbit(w0, dt=dt, t1=0, t2=t_integ)
    w = w[:,0]

    # rotate the model points to stream coordinates
    galactocentric_frame = reference_frame.get('galactocentric_frame', coord.Galactocentric)
    model_c = galactocentric_frame.realize_frame(coord.CartesianRepresentation(w[:,:3].T*u.kpc))\
                                  .transform_to(coord.Galactic)
    model_c_rot = coord.CartesianRepresentation(R.dot(model_c.represent_as(coord.CartesianRepresentation).xyz.value)*u.kpc)\
                       .represent_as(coord.SphericalRepresentation)
    model_phi1 = model_c_rot.lon
    model_phi2 = model_c_rot.lat
    model_d = model_c_rot.distance
    model_mul,model_mub,model_vr = vgal_to_hel(model_c, w[:,3:].T*u.kpc/u.Myr,
                                               vcirc=reference_frame.get('vcirc', None),
                                               vlsr=reference_frame.get('vlsr', None),
                                               galactocentric_frame=galactocentric_frame)

    # rotate the data to stream coordinates
    data_rot_sph = rotate_sph_coordinate(data_coord, R)
    cosphi1_data = np.cos(data_rot_sph.lon).value
    cosphi1_model = np.cos(model_phi1).value
    ix = np.argsort(cosphi1_model)

    # define interpolating functions
    order = 1
    phi2_interp = InterpolatedUnivariateSpline(cosphi1_model[ix], model_phi2[ix].radian, k=order, bbox=[-1,1])
    d_interp = InterpolatedUnivariateSpline(cosphi1_model[ix], model_d[ix].decompose(galactic).value, k=order, bbox=[-1,1])
    mul_interp = InterpolatedUnivariateSpline(cosphi1_model[ix], model_mul[ix].decompose(galactic).value, k=order, bbox=[-1,1])
    mub_interp = InterpolatedUnivariateSpline(cosphi1_model[ix], model_mub[ix].decompose(galactic).value, k=order, bbox=[-1,1])
    vr_interp = InterpolatedUnivariateSpline(cosphi1_model[ix], model_vr[ix].decompose(galactic).value, k=order, bbox=[-1,1])

    chi2 = 0.
    chi2 += -(phi2_interp(cosphi1_data) - data_rot_sph.lat.radian)**2 / (2*phi2_sigma**2)
    chi2 += -(d_interp(cosphi1_data) - data_rot_sph.distance.decompose(galactic).value)**2 / (2*data_uncer[2].decompose(galactic).value**2)
    chi2 += -(mul_interp(cosphi1_data) - data_veloc[0].decompose(galactic).value)**2 / (2*data_uncer[3].decompose(galactic).value**2)
    chi2 += -(mub_interp(cosphi1_data) - data_veloc[1].decompose(galactic).value)**2 / (2*data_uncer[4].decompose(galactic).value**2)
    chi2 += -(vr_interp(cosphi1_data) - data_veloc[2].decompose(galactic).value)**2 / (2*data_uncer[5].decompose(galactic).value**2)

    return chi2

def ln_posterior(p, *args, **kwargs):
    """
    Evaluate the stream orbit fit posterior probability.

    Parameters
    ----------
    p : iterable
        The parameters of the model: the 6 orbital initial conditions, the integration time,
        intrinsic angular width of the stream.
    data_coord : :class:`astropy.coordinate.SkyCoord`, :class:`astropy.coordinate.BaseCoordinateFrame`
    data_veloc : iterable
        An iterable of Astropy :class:`astropy.units.Quantity` objects for the proper motions and
        line-of-sight velocity. The proper motions should be in the same coordinate frame as
        ``data_coord``.
    data_uncer : iterable
        An iterable of Astropy :class:`astropy.units.Quantity` objects for the uncertainties in
        each observable. Should have length = 6.
    potential : :class:`gary.potential.PotentialBase`
        The gravitational potential.
    dt : float
        Timestep for integrating the orbit.
    R : :class:`numpy.ndarray`
        The rotation matrix to convert from the coordinate frame of ``data_coord`` to stream
        coordinates.
    reference_frame : dict (optional)
        Any parameters that specify the reference frame, such as the Sun-Galactic Center distance,
        the circular velocity of the Sun, etc.

    Returns
    -------
    lp : float
        The log of the posterior probability.

    """

    lp = ln_prior(p)
    if not np.isfinite(lp):
        return -np.inf

    ll = ln_likelihood(p, *args, **kwargs)
    if not np.all(np.isfinite(ll)):
        return -np.inf

    return lp + ll.sum()
