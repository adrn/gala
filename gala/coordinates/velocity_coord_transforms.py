# coding: utf-8

""" Velocity transformations for coordinate representations. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library

# Third-party
import numpy as np
import astropy.coordinates as coord

__all__ = ['cartesian_to_spherical', 'cartesian_to_physicsspherical',
           'cartesian_to_cylindrical', 'spherical_to_cartesian',
           'physicsspherical_to_cartesian', 'cylindrical_to_cartesian']

def _pos_to_repr(pos):

    if hasattr(pos, 'xyz'):  # Representation-like
        pos_repr = coord.CartesianRepresentation(pos.xyz)

    elif hasattr(pos, 'cartesian') or hasattr(pos, 'to_cartesian'):  # Frame-like
        pos_repr = pos.represent_as(coord.CartesianRepresentation)

    elif hasattr(pos, 'unit'):  # Quantity-like
        pos_repr = coord.CartesianRepresentation(pos)

    else:
        raise TypeError("Unsupported position type '{0}'. Position must be "
                        "an Astropy Representation or Frame, or a "
                        "Quantity instance".format(type(pos)))

    return pos_repr

def cartesian_to_spherical(pos, vel):
    r"""
    Convert a velocity in Cartesian coordinates to velocity components
    in spherical coordinates. This follows the naming convention used in
    :mod:`astropy.coordinates`: spherical coordinates consist of a distance,
    :math:`d`, a longitude, :math:`\phi`, in the range [0, 360] deg, and
    a latitude, :math:`b`, in the range [-90, 90] deg.

    The components of the output velocity all have units of velocity, i.e.,
    this is not used for transforming from a cartesian velocity to angular
    velocities, but rather the velocity vector components in Eq. 2 below.

    .. math::

        \boldsymbol{v} &= v_x\boldsymbol{\hat{x}} + v_y\boldsymbol{\hat{y}} + v_z\boldsymbol{\hat{z}}\\\\
        &= v_r\boldsymbol{\hat{d}} + v_\phi\boldsymbol{\hat{\phi}} + v_b\boldsymbol{\hat{b}}\\\\
        &= \dot{d}\boldsymbol{\hat{d}} + d\cos b \dot{\phi}\boldsymbol{\hat{\phi}} + d\dot{b}\boldsymbol{\hat{b}}

    Parameters
    ----------
    pos : :class:`~astropy.units.Quantity`, :class:`~astropy.coordinates.BaseCoordinateFrame`, :class:`~astropy.coordinates.BaseRepresentation`
        Input position or positions as one of the allowed types. You may pass in a
        :class:`~astropy.units.Quantity` with :class:`~astropy.units.dimensionless_unscaled`
        units if you are working in natural units.
    vel : :class:`~astropy.units.Quantity`
        Input velocity or velocities as one of the allowed types. You may pass in a
        :class:`~astropy.units.Quantity` with :class:`~astropy.units.dimensionless_unscaled`
        units if you are working in natural units. axis=0 is assumed to be the
        dimensionality axis, e.g., ``vx,vy,vz = vel`` should work.

    Returns
    -------
    vsph : :class:`~astropy.units.Quantity`
        Array of spherical velocity components. Will have the same shape as the
        input velocity.

    Examples
    --------

    """

    # position in Cartesian and spherical
    car_pos = _pos_to_repr(pos)
    sph_pos = car_pos.represent_as(coord.SphericalRepresentation)

    if not hasattr(vel, 'unit'):
        raise TypeError("Unsupported velocity type '{}'. Velocity must be "
                        "an Astropy Quantity instance.".format(type(vel)))

    # get out spherical components
    d = sph_pos.distance
    dxy = np.sqrt(car_pos.x**2 + car_pos.y**2)

    vr = np.sum(car_pos.xyz * vel, axis=0) / d

    mu_lon = (car_pos.xyz[0]*vel[1] - vel[0]*car_pos.xyz[1]) / dxy**2
    vlon = mu_lon * d * np.cos(sph_pos.lat)

    mu_lat = (car_pos.xyz[2]*(car_pos.xyz[0]*vel[0] + car_pos.xyz[1]*vel[1]) - dxy**2*vel[2]) / d**2 / dxy
    vlat = -mu_lat * d

    vsph = np.zeros_like(vel)
    vsph[0] = vr
    vsph[1] = vlon
    vsph[2] = vlat
    return vsph

def cartesian_to_physicsspherical(pos, vel):
    r"""
    Convert a velocity in Cartesian coordinates to velocity components
    in spherical coordinates. This follows the naming convention used in
    :mod:`astropy.coordinates`: physics spherical coordinates consist of
    a radius, :math:`r`, a longitude, :math:`\phi`, in the range [0, 360]
    deg and a colatitude, :math:`\theta`, in the range [0, 180] deg,
    measured from the z-axis.

    The components of the output velocity all have units of velocity, i.e.,
    this is not used for transforming from a cartesian velocity to angular
    velocities, but rather the velocity vector components in Eq. 2 below.

    .. math::

        \boldsymbol{v} &= v_x\boldsymbol{\hat{x}} + v_y\boldsymbol{\hat{y}} + v_z\boldsymbol{\hat{z}}\\\\
        &= v_r\boldsymbol{\hat{r}} + v_\phi\boldsymbol{\hat{\phi}} + v_\theta\boldsymbol{\hat{\theta}}\\\\
        &= \dot{r}\boldsymbol{\hat{r}} + r\sin\theta \dot{\phi}\boldsymbol{\hat{\phi}} + r\dot{\theta}\boldsymbol{\hat{\theta}}

    Parameters
    ----------
    pos : :class:`~astropy.units.Quantity`, :class:`~astropy.coordinates.BaseCoordinateFrame`, :class:`~astropy.coordinates.BaseRepresentation`
        Input position or positions as one of the allowed types. You may pass in a
        :class:`~astropy.units.Quantity` with :class:`~astropy.units.dimensionless_unscaled`
        units if you are working in natural units.
    vel : :class:`~astropy.units.Quantity`
        Input velocity or velocities as one of the allowed types. You may pass in a
        :class:`~astropy.units.Quantity` with :class:`~astropy.units.dimensionless_unscaled`
        units if you are working in natural units. axis=0 is assumed to be the
        dimensionality axis, e.g., ``vx,vy,vz = vel`` should work.

    Returns
    -------
    vsph : :class:`~astropy.units.Quantity`
        Array of spherical velocity components. Will have the same shape as the
        input velocity.

    Examples
    --------

    """

    # position in Cartesian and spherical
    car_pos = _pos_to_repr(pos)
    sph_pos = car_pos.represent_as(coord.PhysicsSphericalRepresentation)

    if not hasattr(vel, 'unit'):
        raise TypeError("Unsupported velocity type '{}'. Velocity must be "
                        "an Astropy Quantity instance.".format(type(vel)))

    # get out spherical components
    r = sph_pos.r
    dxy = np.sqrt(car_pos.x**2 + car_pos.y**2)

    vr = np.sum(car_pos.xyz * vel, axis=0) / r

    mu_lon = (car_pos.xyz[0]*vel[1] - vel[0]*car_pos.xyz[1]) / dxy**2
    vlon = mu_lon * r * np.sin(sph_pos.theta)

    mu_lat = (car_pos.xyz[2]*(car_pos.xyz[0]*vel[0] + car_pos.xyz[1]*vel[1]) - dxy**2*vel[2]) / r**2 / dxy
    vlat = mu_lat * r

    vsph = np.zeros_like(vel)
    vsph[0] = vr
    vsph[1] = vlon
    vsph[2] = vlat
    return vsph

def cartesian_to_cylindrical(pos, vel):
    r"""
    Convert a velocity in Cartesian coordinates to velocity components
    in cylindrical coordinates. This follows the naming convention used in
    :mod:`astropy.coordinates`: cylindrical coordinates consist of a radius,
    :math:`\rho`, an azimuthal angle, :math:`\phi`, in the range [0, 360] deg,
    and a z coordinate, :math:`z`.

    The components of the output velocity all have units of velocity, i.e.,
    this is not used for transforming from a cartesian velocity to angular
    velocities, but rather the velocity vector components in Eq. 2 below.

    .. math::

        \boldsymbol{v} &= v_x\boldsymbol{\hat{x}} + v_y\boldsymbol{\hat{y}} + v_z\boldsymbol{\hat{z}}\\\\
        &= v_\rho\boldsymbol{\hat{\rho}} + v_\phi\boldsymbol{\hat{\phi}} + v_z\boldsymbol{\hat{z}}\\\\
        &= \dot{\rho}\boldsymbol{\hat{\rho}} + \rho\dot{\phi}\boldsymbol{\hat{\phi}} + \dot{z}\boldsymbol{\hat{\theta}}

    Parameters
    ----------
    pos : :class:`~astropy.units.Quantity`, :class:`~astropy.coordinates.BaseCoordinateFrame`, :class:`~astropy.coordinates.BaseRepresentation`
        Input position or positions as one of the allowed types. You may pass in a
        :class:`~astropy.units.Quantity` with :class:`~astropy.units.dimensionless_unscaled`
        units if you are working in natural units.
    vel : :class:`~astropy.units.Quantity`
        Input velocity or velocities as one of the allowed types. You may pass in a
        :class:`~astropy.units.Quantity` with :class:`~astropy.units.dimensionless_unscaled`
        units if you are working in natural units. axis=0 is assumed to be the
        dimensionality axis, e.g., ``vx,vy,vz = vel`` should work.

    Returns
    -------
    vcyl : :class:`~astropy.units.Quantity`
        Array of spherical velocity components. Will have the same shape as the
        input velocity.

    Examples
    --------

    """

    # position in Cartesian and spherical
    car_pos = _pos_to_repr(pos)
    cyl_pos = car_pos.represent_as(coord.CylindricalRepresentation)

    if not hasattr(vel, 'unit'):
        raise TypeError("Unsupported velocity type '{}'. Velocity must be "
                        "an Astropy Quantity instance.".format(type(vel)))

    # get out spherical components
    rho = cyl_pos.rho
    vrho = np.sum(car_pos.xyz[:2] * vel[:2], axis=0) / rho

    phidot = (car_pos.xyz[0]*vel[1] - vel[0]*car_pos.xyz[1]) / rho**2
    vphi = phidot * rho

    vcyl = np.zeros_like(vel)
    vcyl[0] = vrho
    vcyl[1] = vphi
    vcyl[2] = vel[2]
    return vcyl

def spherical_to_cartesian(pos, vel):
    r"""
    Convert a velocity in Spherical coordinates to Cartesian coordinates.
    This follows the naming convention used in :mod:`astropy.coordinates`:
    spherical coordinates consist of a distance, :math:`d`, a longitude,
    :math:`\phi`, in the range [0, 360] deg, and a latitude, :math:`b`,
    in the range [-90, 90] deg.

    All components of the input spherical velocity should have units of
    velocity, i.e., this is not used for transforming angular velocities,
    but rather the velocity vector components in Eq. 2 below.

    .. math::

        \boldsymbol{v} &= v_x\boldsymbol{\hat{x}} + v_y\boldsymbol{\hat{y}} + v_z\boldsymbol{\hat{z}}\\\\
        &= v_r\boldsymbol{\hat{d}} + v_\phi\boldsymbol{\hat{\phi}} + v_b\boldsymbol{\hat{b}}\\\\
        &= \dot{d}\boldsymbol{\hat{d}} + d\cos b \dot{\phi}\boldsymbol{\hat{\phi}} + d\dot{b}\boldsymbol{\hat{b}}

    Parameters
    ----------
    pos : :class:`~astropy.units.Quantity`, :class:`~astropy.coordinates.BaseCoordinateFrame`, :class:`~astropy.coordinates.BaseRepresentation`
        Input position or positions as one of the allowed types. You may pass in a
        :class:`~astropy.units.Quantity` with :class:`~astropy.units.dimensionless_unscaled`
        units if you are working in natural units.
    vel : :class:`~astropy.units.Quantity`
        Input velocity or velocities as one of the allowed types. You may pass in a
        :class:`~astropy.units.Quantity` with :class:`~astropy.units.dimensionless_unscaled`
        units if you are working in natural units. axis=0 is assumed to be the
        dimensionality axis, e.g., ``vx,vy,vz = vel`` should work.

    Returns
    -------
    vxyz : :class:`~astropy.units.Quantity`
        Array of Cartesian velocity components. Will have the same shape as the
        input velocity.

    Examples
    --------

    """

    # position in Cartesian and spherical
    car_pos = _pos_to_repr(pos)
    sph_pos = car_pos.represent_as(coord.SphericalRepresentation)

    if not hasattr(vel, 'unit'):
        raise TypeError("Unsupported velocity type '{}'. Velocity must be "
                        "an Astropy Quantity instance.".format(type(vel)))

    # get out spherical components
    phi = sph_pos.lon
    lat = sph_pos.lat
    vr,vlon,vlat = vel

    vxyz = np.zeros_like(vel)
    vxyz[0] = vr*np.cos(phi)*np.cos(lat) - np.sin(phi)*vlon - np.cos(phi)*np.sin(lat)*vlat
    vxyz[1] = vr*np.sin(phi)*np.cos(lat) + np.cos(phi)*vlon - np.sin(phi)*np.sin(lat)*vlat
    vxyz[2] = vr*np.sin(lat) + np.cos(lat)*vlat
    return vxyz

def physicsspherical_to_cartesian(pos, vel):
    r"""
    Convert a velocity in Spherical coordinates to Cartesian coordinates.
    This follows the naming convention used in :mod:`astropy.coordinates`:
    physics spherical coordinates consist of a radius, :math:`r`, a longitude,
    :math:`\phi`, in the range [0, 360] deg and a colatitude, :math:`\theta`,
    in the range [0, 180] deg, measured from the z-axis.

    The components of the output velocity all have units of velocity, i.e.,
    this is not used for transforming from a cartesian velocity to angular
    velocities, but rather the velocity vector components in Eq. 2 below.

    .. math::

        \boldsymbol{v} &= v_x\boldsymbol{\hat{x}} + v_y\boldsymbol{\hat{y}} + v_z\boldsymbol{\hat{z}}\\\\
        &= v_r\boldsymbol{\hat{r}} + v_\phi\boldsymbol{\hat{\phi}} + v_\theta\boldsymbol{\hat{\theta}}\\\\
        &= \dot{r}\boldsymbol{\hat{r}} + r\sin\theta \dot{\phi}\boldsymbol{\hat{\phi}} + r\dot{\theta}\boldsymbol{\hat{\theta}}

    Parameters
    ----------
    pos : :class:`~astropy.units.Quantity`, :class:`~astropy.coordinates.BaseCoordinateFrame`, :class:`~astropy.coordinates.BaseRepresentation`
        Input position or positions as one of the allowed types. You may pass in a
        :class:`~astropy.units.Quantity` with :class:`~astropy.units.dimensionless_unscaled`
        units if you are working in natural units.
    vel : :class:`~astropy.units.Quantity`
        Input velocity or velocities as one of the allowed types. You may pass in a
        :class:`~astropy.units.Quantity` with :class:`~astropy.units.dimensionless_unscaled`
        units if you are working in natural units. axis=0 is assumed to be the
        dimensionality axis, e.g., ``vx,vy,vz = vel`` should work.

    Returns
    -------
    vxyz : :class:`~astropy.units.Quantity`
        Array of Cartesian velocity components. Will have the same shape as the
        input velocity.

    Examples
    --------

    """

    # position in Cartesian and spherical
    car_pos = _pos_to_repr(pos)
    sph_pos = car_pos.represent_as(coord.PhysicsSphericalRepresentation)

    if not hasattr(vel, 'unit'):
        raise TypeError("Unsupported velocity type '{}'. Velocity must be "
                        "an Astropy Quantity instance.".format(type(vel)))

    # get out spherical components
    phi = sph_pos.phi
    the = sph_pos.theta
    vr,vphi,vthe = vel
    vthe = -vthe

    vxyz = np.zeros_like(vel)
    vxyz[0] = vr*np.cos(phi)*np.sin(the) - np.sin(phi)*vphi - np.cos(phi)*np.cos(the)*vthe
    vxyz[1] = vr*np.sin(phi)*np.sin(the) + np.cos(phi)*vphi - np.sin(phi)*np.cos(the)*vthe
    vxyz[2] = vr*np.cos(the) + np.sin(the)*vthe
    return vxyz

def cylindrical_to_cartesian(pos, vel):
    r"""
    Convert a velocity in Spherical coordinates to Cylindrical coordinates.
    This follows the naming convention used in :mod:`astropy.coordinates`:
    cylindrical coordinates consist of a radius, :math:`\rho`, an azimuthal angle,
    :math:`\phi`, in the range [0, 360] deg, and a z coordinate, :math:`z`.

    The components of the output velocity all have units of velocity, i.e.,
    this is not used for transforming from a cartesian velocity to angular
    velocities, but rather the velocity vector components in Eq. 2 below.

    .. math::

        \boldsymbol{v} &= v_x\boldsymbol{\hat{x}} + v_y\boldsymbol{\hat{y}} + v_z\boldsymbol{\hat{z}}\\\\
        &= v_\rho\boldsymbol{\hat{\rho}} + v_\phi\boldsymbol{\hat{\phi}} + v_z\boldsymbol{\hat{z}}\\\\
        &= \dot{\rho}\boldsymbol{\hat{\rho}} + \rho\dot{\phi}\boldsymbol{\hat{\phi}} + \dot{z}\boldsymbol{\hat{\theta}}

    Parameters
    ----------
    pos : :class:`~astropy.units.Quantity`, :class:`~astropy.coordinates.BaseCoordinateFrame`, :class:`~astropy.coordinates.BaseRepresentation`
        Input position or positions as one of the allowed types. You may pass in a
        :class:`~astropy.units.Quantity` with :class:`~astropy.units.dimensionless_unscaled`
        units if you are working in natural units.
    vel : :class:`~astropy.units.Quantity`
        Input velocity or velocities as one of the allowed types. You may pass in a
        :class:`~astropy.units.Quantity` with :class:`~astropy.units.dimensionless_unscaled`
        units if you are working in natural units. axis=0 is assumed to be the
        dimensionality axis, e.g., ``vx,vy,vz = vel`` should work.

    Returns
    -------
    vxyz : :class:`~astropy.units.Quantity`
        Array of Cartesian velocity components. Will have the same shape as the
        input velocity.

    Examples
    --------

    """

    # position in Cartesian and cylindrical
    car_pos = _pos_to_repr(pos)
    cyl_pos = car_pos.represent_as(coord.CylindricalRepresentation)

    if not hasattr(vel, 'unit'):
        raise TypeError("Unsupported velocity type '{}'. Velocity must be "
                        "an Astropy Quantity instance.".format(type(vel)))

    # get out spherical components
    phi = cyl_pos.phi
    vrho,vphi,vz = vel

    vcyl = np.zeros_like(vel)
    vcyl[0] = vrho * np.cos(phi) - np.sin(phi) * vphi
    vcyl[1] = vrho * np.sin(phi) + np.cos(phi) * vphi
    vcyl[2] = vz
    return vcyl
