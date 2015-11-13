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
from scipy.stats import norm

# Project
from ..coordinates import Quaternion, vgal_to_hel, vhel_to_gal
from ..units import galactic
from ..integrate import DOPRI853Integrator

__all__ = ['compute_stream_rotation_matrix', 'rotate_sph_coordinate',
           'ln_prior', 'ln_likelihood', 'ln_posterior']

def _rotation_opt_func(qua_wxyz, xyz):
    """
    Given a quaternion vector ``(w,x,y,z)`` and the data in heliocentric,
    Cartesian coordinates, compute the cost function for the given rotation.
    """
    R = Quaternion(qua_wxyz).rotation_matrix
    sph = coord.CartesianRepresentation(R.dot(xyz)*uno)\
               .represent_as(coord.SphericalRepresentation)
    return np.sum(sph.lat.degree**2)

def compute_stream_rotation_matrix(coordinate, wxyz0=None, align_lon='mean'):
    """
    Compute the rotation matrix to go from the frame of the input
    coordinate to closely align the equator with the stream.

    Parameter
    ---------
    coordinate : :class:`astropy.coordinate.SkyCoord`, :class:`astropy.coordinate.BaseCoordinateFrame`
        The coordinates of the stream stars.
    wxyz0 : array_like (optional)
        Initial guess for the quaternion vector that represents the rotation.
    align_lon : str, int (optional)
        Can specify either 'min', 'max', or an integer index. This picks the
        'pirvot' star, whose longitude is set to 0.

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

    new_xyz = R.dot(coordinate.cartesian.xyz.value)
    lon = np.arctan2(new_xyz[1], new_xyz[0])
    if align_lon == 'mean':
        _lon = np.mean(lon)
        R3 = 1.
    elif align_lon == 'min':
        ix = lon.argmin()
        _lon = lon[ix]
        R3 = 1.
    elif align_lon == 'max':
        ix = lon.argmax()
        _lon = lon[ix]
        R3 = rotation_matrix(np.pi*u.radian, 'x')
    else:
        ix = int(align_lon)
        _lon = lon[ix]
        R3 = 1.
    R2 = rotation_matrix(_lon*u.radian, 'z')
    R = R3*R2*R

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

def ln_prior(p, data_coord, data_veloc, data_uncer, potential, dt, R, reference_frame=dict(),
             fix_phi2_sigma=False, fix_d_sigma=False, fix_vr_sigma=False):
    """
    Evaluate the prior over stream orbit fit parameters.

    See docstring for `ln_likelihood()` for information on args and kwargs.

    """
    # these are for the initial conditions
    phi2,d,mul,mub,vr = p[:5]
    t_integ = p[5]

    lp = 0.

    # prior on instrinsic width of stream
    if not fix_phi2_sigma:
        phi2_sigma = p[6] # intrinsic width on sky
        if phi2_sigma <= 0.:
            return -np.inf
        lp += -np.log(phi2_sigma)
    else:
        phi2_sigma = fix_phi2_sigma

    # prior on instrinsic depth of stream
    if not fix_d_sigma:
        d_sigma = p[7] # intrinsic depth (in distance)
        if d_sigma <= 0.:
            return -np.inf
        lp += -np.log(d_sigma)
    else:
        d_sigma = fix_d_sigma

    # prior on instrinsic LOS velocity dispersion of stream
    if not fix_vr_sigma:
        vr_sigma = p[8]
        if vr_sigma <= 0.:
            return -np.inf
        lp += -np.log(vr_sigma)
    else:
        vr_sigma = fix_vr_sigma

    # strong prior on phi2
    if phi2 < -np.pi/2. or phi2 > np.pi/2:
        return -np.inf
    lp += norm.logpdf(phi2, loc=0., scale=phi2_sigma)

    # uniform prior on integration time
    ntimes = int(t_integ / dt) + 1
    if np.sign(dt)*t_integ <= 1. or np.sign(dt)*t_integ > 1000. or ntimes < 4:
        return -np.inf

    return lp

def ln_likelihood(p, data_coord, data_veloc, data_uncer, potential, dt, R, reference_frame=dict(),
                  fix_phi2_sigma=False, fix_d_sigma=False, fix_vr_sigma=False):
    """
    Evaluate the stream orbit fit likelihood.

    Parameters
    ----------
    p : iterable
        The parameters of the model: distance, proper motions, radial velocity, the integration time,
        and (optionally) intrinsic angular width of the stream.
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
    chi2 = 0.

    # the Galactocentric frame we're using
    gc_frame = reference_frame.get('galactocentric_frame', coord.Galactocentric())

    # read in from the parameter vector to variables
    # these are for the initial conditions
    phi2,d,mul,mub,vr = p[:5]
    phi2 *= u.radian
    d *= u.kpc
    mul *= u.radian/u.Myr
    mub *= u.radian/u.Myr
    vr *= u.kpc/u.Myr

    t_integ = p[5]
    if not fix_phi2_sigma:
        phi2_sigma = p[6] # intrinsic width on sky
    else:
        phi2_sigma = fix_phi2_sigma

    if not fix_d_sigma:
        d_sigma = p[7] # intrinsic width in distance
    else:
        d_sigma = fix_d_sigma

    if not fix_vr_sigma:
        vr_sigma = p[8] # intrinsic LOS velocity dispersion
    else:
        vr_sigma = fix_vr_sigma

    # convert initial conditions from stream coordinates to data coordinate frame
    sph = coord.SphericalRepresentation(lon=0.*u.radian, lat=phi2, distance=d)
    xyz = sph.represent_as(coord.CartesianRepresentation).xyz.value
    in_frame_car = coord.CartesianRepresentation(R.T.dot(xyz).T*u.kpc)
    initial_coord = data_coord.realize_frame(in_frame_car)

    # now convert to galactocentric coordinates
    x0 = initial_coord.transform_to(gc_frame).cartesian.xyz.decompose(galactic).value
    v0 = vhel_to_gal(initial_coord, pm=(mul,mub), rv=vr,
                     **reference_frame).decompose(galactic).value
    w0 = np.append(x0, v0)

    # HACK: a prior on velocities
    vmag2 = np.sum(v0**2)
    chi2 += -vmag2 / (0.15**2)

    # integrate the orbit
    t,w = potential.integrate_orbit(w0, dt=np.sign(t_integ)*np.abs(dt), t1=0, t2=t_integ,
                                    Integrator=DOPRI853Integrator)
    w = w[:,0]

    # rotate the model points to stream coordinates
    model_c = gc_frame.realize_frame(coord.CartesianRepresentation(w[:,:3].T*u.kpc))\
                      .transform_to(coord.Galactic)
    model_c_rot = coord.CartesianRepresentation(R.dot(model_c.represent_as(coord.CartesianRepresentation).xyz.value)*u.kpc)\
                       .represent_as(coord.SphericalRepresentation)
    model_phi1 = model_c_rot.lon
    model_phi2 = model_c_rot.lat
    model_d = model_c_rot.distance
    model_mul,model_mub,model_vr = vgal_to_hel(model_c, w[:,3:].T*u.kpc/u.Myr, **reference_frame)

    # rotate the data to stream coordinates
    data_rot_sph = rotate_sph_coordinate(data_coord, R)
    cosphi1_data = np.cos(data_rot_sph.lon).value
    cosphi1_model = np.cos(model_phi1).value
    ix = np.argsort(cosphi1_model)

    # define interpolating functions
    order = 3
    phi2_interp = InterpolatedUnivariateSpline(cosphi1_model[ix], model_phi2[ix].radian, k=order, bbox=[-1,1])
    d_interp = InterpolatedUnivariateSpline(cosphi1_model[ix], model_d[ix].decompose(galactic).value, k=order, bbox=[-1,1])
    mul_interp = InterpolatedUnivariateSpline(cosphi1_model[ix], model_mul[ix].decompose(galactic).value, k=order, bbox=[-1,1])
    mub_interp = InterpolatedUnivariateSpline(cosphi1_model[ix], model_mub[ix].decompose(galactic).value, k=order, bbox=[-1,1])
    vr_interp = InterpolatedUnivariateSpline(cosphi1_model[ix], model_vr[ix].decompose(galactic).value, k=order, bbox=[-1,1])

    chi2 += -(phi2_interp(cosphi1_data) - data_rot_sph.lat.radian)**2 / (phi2_sigma**2) - 2*np.log(phi2_sigma)

    err = data_uncer[2].decompose(galactic).value
    chi2 += -(d_interp(cosphi1_data) - data_rot_sph.distance.decompose(galactic).value)**2 / (err**2 + d_sigma**2) - np.log(err**2 + d_sigma**2)

    err = data_uncer[5].decompose(galactic).value
    chi2 += -(vr_interp(cosphi1_data) - data_veloc[2].decompose(galactic).value)**2 / (err**2 + vr_sigma**2) - np.log(err**2 + vr_sigma**2)

    for i,interp in enumerate([mul_interp, mub_interp]):
        err = data_uncer[3+i].decompose(galactic).value
        chi2 += -(interp(cosphi1_data) - data_veloc[i].decompose(galactic).value)**2 / (err**2) - 2*np.log(err)

    # this is some kind of whack prior - don't integrate more than we have to
    # chi2 += -(model_phi1.radian.min() - data_rot_sph.lon.radian.min())**2 / ((2*phi2_sigma)**2)
    # chi2 += -(model_phi1.radian.max() - data_rot_sph.lon.radian.max())**2 / ((2*phi2_sigma)**2)

    return 0.5*chi2

def ln_posterior(p, *args, **kwargs):
    """
    Evaluate the stream orbit fit posterior probability.

    See docstring for `ln_likelihood()` for information on args and kwargs.

    Returns
    -------
    lp : float
        The log of the posterior probability.

    """

    lp = ln_prior(p, *args, **kwargs)
    if not np.isfinite(lp):
        return -np.inf

    ll = ln_likelihood(p, *args, **kwargs)
    if not np.all(np.isfinite(ll)):
        return -np.inf

    return lp + ll.sum()
