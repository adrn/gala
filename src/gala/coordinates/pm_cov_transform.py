import astropy.coordinates as coord
import numpy as np

__all__ = ["transform_pm_cov"]


def get_uv_tan(c):
    """Get tangent plane basis vectors on the unit sphere at the given
    spherical coordinates.
    """
    l = c.spherical.lon
    b = c.spherical.lat

    p = np.array([-np.sin(l), np.cos(l), np.zeros_like(l.value)]).T
    q = np.array([-np.cos(l) * np.sin(b), -np.sin(l) * np.sin(b), np.cos(b)]).T

    return np.stack((p, q), axis=-1)


def get_transform_matrix(from_frame, to_frame):
    """Compose sequential matrix transformations (static or dynamic) to get a
    single transformation matrix from a given path through the Astropy
    transformation machinery.

    Parameters
    ----------
    from_frame : `~astropy.coordinates.BaseCoordinateFrame` subclass
        The *class* or instance of the frame you're transforming from.
    to_frame : `~astropy.coordinates.BaseCoordinateFrame` subclass
        The class or instance of the frame you're transforming to.
    """
    if isinstance(from_frame, coord.BaseCoordinateFrame):
        from_frame_cls = from_frame.__class__
    else:
        from_frame_cls = from_frame

    if isinstance(to_frame, coord.BaseCoordinateFrame):
        to_frame_cls = to_frame.__class__
    else:
        to_frame_cls = to_frame

    path, _distance = coord.frame_transform_graph.find_shortest_path(
        from_frame_cls, to_frame_cls
    )

    matrices = []
    currsys = from_frame
    for p in path[1:]:  # first element is fromsys so we skip it
        if isinstance(currsys, coord.BaseCoordinateFrame):
            currsys_cls = currsys.__class__
        else:
            currsys_cls = currsys
            currsys = currsys_cls()

        trans = coord.frame_transform_graph._graph[currsys_cls][p]

        if isinstance(to_frame, p):
            p = to_frame

        if isinstance(trans, coord.DynamicMatrixTransform):
            if not isinstance(p, coord.BaseCoordinateFrame):
                p = p()
            M = trans.matrix_func(currsys, p)
        elif isinstance(trans, coord.StaticMatrixTransform):
            M = trans.matrix
        else:
            msg = (
                f"Transform path contains a '{trans.__class__.__name__}': cannot "
                "be composed into a single transformation "
                "matrix."
            )
            raise ValueError(msg)

        matrices.append(M)
        currsys = p

    M = None
    for Mi in reversed(matrices):
        M = Mi if M is None else M @ Mi

    return M


def transform_pm_cov(c, cov, to_frame):
    """Transform a proper motion covariance matrix to a new frame.

    Parameters
    ----------
    c : `~astropy.coordinates.SkyCoord`
        The sky coordinates of the sources in the initial coordinate frame.
    cov : array_like
        The covariance matrix of the proper motions. Must have same length as
        the input coordinates.
    to_frame : `~astropy.coordinates.BaseCoordinateFrame` subclass
        The frame to transform to as an Astropy coordinate frame class or
        instance.

    Returns
    -------
    new_cov : array_like
        The transformed covariance matrix.

    """
    if c.isscalar and cov.shape != (2, 2):
        msg = (
            "If input coordinate object is a scalar coordinate, "
            "the proper motion covariance matrix must have shape "
            f"(2, 2), not {cov.shape}"
        )
        raise ValueError(msg)

    if not c.isscalar and len(c) != cov.shape[0]:
        msg = (
            "Input coordinates and covariance matrix must have "
            f"the same number of entries ({len(c)} vs {cov.shape[0]})."
        )
        raise ValueError(msg)

    # 3D rotation matrix, to be projected onto the tangent plane
    frame = c.frame if hasattr(c, "frame") else c
    R = get_transform_matrix(frame.__class__, to_frame)

    # Get input coordinates in the desired frame:
    c_to = c.transform_to(to_frame)

    # Get tangent plane coordinates:
    uv_in = get_uv_tan(c)
    uv_to = get_uv_tan(c_to)

    if not c.isscalar:
        G = np.einsum("nab, nac->nbc", uv_to, np.einsum("ji, nik->njk", R, uv_in))

        # transform
        cov_to = np.einsum("nba, nac->nbc", G, np.einsum("nij, nkj->nik", cov, G))
    else:
        G = np.einsum("ab, ac->bc", uv_to, np.einsum("ji, ik->jk", R, uv_in))

        # transform
        cov_to = np.einsum("ba, ac->bc", G, np.einsum("ij, kj->ik", cov, G))

    return cov_to
