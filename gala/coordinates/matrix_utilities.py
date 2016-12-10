# -*- coding: utf-8 -*-
# All code below except matmul() is licensed under a 3-clause BSD style license, included in the
# header below.
#
#
# - This module is taken from the current v1.3.x branch of Astropy and should be
#   deleted from Gala once we required Astropy v1.3 (currently unreleased).
#
# Astropy license:
# Copyright (c) 2011-2016, Astropy Developers

# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
# * Neither the name of the Astropy Team nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Numpy license:
# Copyright (c) 2005-2011, NumPy Developers.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.

#     * Neither the name of the NumPy Developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
This module contains utililies used for constructing rotation matrices.
"""
from functools import reduce
import numpy as np

import astropy.units as u
from astropy.coordinates.angles import Angle
from astropy.extern.six.moves import range

def matmul(a, b, out=None):
        """Matrix product of two arrays.
        The behavior depends on the arguments in the following way.
        - If both arguments are 2-D they are multiplied like conventional
          matrices.
        - If either argument is N-D, N > 2, it is treated as a stack of
          matrices residing in the last two indexes and broadcast accordingly.
        - If the first argument is 1-D, it is promoted to a matrix by
          prepending a 1 to its dimensions. After matrix multiplication
          the prepended 1 is removed.
        - If the second argument is 1-D, it is promoted to a matrix by
          appending a 1 to its dimensions. After matrix multiplication
          the appended 1 is removed.
        Multiplication by a scalar is not allowed, use ``*`` instead. Note that
        multiplying a stack of matrices with a vector will result in a stack of
        vectors, but matmul will not recognize it as such.
        ``matmul`` differs from ``dot`` in two important ways.
        - Multiplication by scalars is not allowed.
        - Stacks of matrices are broadcast together as if the matrices
          were elements.
        Parameters
        ----------
        a : array_like
            First argument.
        b : array_like
            Second argument.
        out : ndarray, optional
            Output argument. This must have the exact kind that would be returned
            if it was not used. In particular, it must have the right type, must be
            C-contiguous, and its dtype must be the dtype that would be returned
            for `dot(a,b)`. This is a performance feature. Therefore, if these
            conditions are not met, an exception is raised, instead of attempting
        Notes
        -----
        This routine mimicks ``matmul`` using ``einsum``.  See
        http://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html
        """
        a = np.asanyarray(a)
        b = np.asanyarray(b)

        if out is None:
            kwargs = {}
        else:
            kwargs = {'out': out}

        if a.ndim >= 2:
            if b.ndim >= 2:
                return np.einsum('...ij,...jk->...ik', a, b, **kwargs)

            if b.ndim == 1:
                return np.einsum('...ij,...j->...i', a, b, **kwargs)

        elif a.ndim == 1 and b.ndim >= 2:
            return np.einsum('...i,...ik->...k', a, b, **kwargs)

        raise ValueError("Scalar operands are not allowed, use '*' instead.")

def matrix_product(*matrices):
    """Matrix multiply all arguments together.
    Arguments should have dimension 2 or larger. Larger dimensional objects
    are interpreted as stacks of matrices residing in the last two dimensions.
    This function mostly exists for readability: using `~numpy.matmul`
    directly, one would have ``matmul(matmul(m1, m2), m3)``, etc. For even
    better readability, one might consider using `~numpy.matrix` for the
    arguments (so that one could write ``m1 * m2 * m3``), but then it is not
    possible to handle stacks of matrices. Once only python >=3.5 is supported,
    this function can be replaced by ``m1 @ m2 @ m3``.
    """
    return reduce(matmul, matrices)


def matrix_transpose(matrix):
    """Transpose a matrix or stack of matrices by swapping the last two axes.
    This function mostly exists for readability; seeing ``.swapaxes(-2, -1)``
    it is not that obvious that one does a transpose.  Note that one cannot
    use `~numpy.ndarray.T`, as this transposes all axes and thus does not
    work for stacks of matrices.
    """
    return matrix.swapaxes(-2, -1)


def rotation_matrix(angle, axis='z', unit=None):
    """
    Generate matrices for rotation by some angle around some axis.
    Parameters
    ----------
    angle : convertible to `Angle`
        The amount of rotation the matrices should represent.  Can be an array.
    axis : str, or array-like
        Either ``'x'``, ``'y'``, ``'z'``, or a (x,y,z) specifying the axis to
        rotate about. If ``'x'``, ``'y'``, or ``'z'``, the rotation sense is
        counterclockwise looking down the + axis (e.g. positive rotations obey
        left-hand-rule).  If given as an array, the last dimension should be 3;
        it will be broadcast against ``angle``.
    unit : UnitBase, optional
        If ``angle`` does not have associated units, they are in this
        unit.  If neither are provided, it is assumed to be degrees.
    Returns
    -------
    rmat : `numpy.matrix`
        A unitary rotation matrix.
    """
    if unit is None:
        unit = u.degree

    angle = Angle(angle, unit=unit)

    s = np.sin(angle)
    c = np.cos(angle)

    # use optimized implementations for x/y/z
    try:
        i = 'xyz'.index(axis)
    except TypeError:
        axis = np.asarray(axis)
        axis = axis / np.sqrt((axis * axis).sum(axis=-1, keepdims=True))
        R = (axis[..., np.newaxis] * axis[..., np.newaxis, :] *
             (1. - c)[..., np.newaxis, np.newaxis])

        for i in range(0, 3):
            R[..., i, i] += c
            a1 = (i + 1) % 3
            a2 = (i + 2) % 3
            R[..., a1, a2] += axis[..., i] * s
            R[..., a2, a1] -= axis[..., i] * s

    else:
        a1 = (i + 1) % 3
        a2 = (i + 2) % 3
        R = np.zeros(angle.shape + (3, 3))
        R[..., i, i] = 1.
        R[..., a1, a1] = c
        R[..., a1, a2] = s
        R[..., a2, a1] = -s
        R[..., a2, a2] = c

    return R


def angle_axis(matrix):
    """
    Angle of rotation and rotation axis for a given rotation matrix.
    Parameters
    ----------
    matrix : array-like
        A 3 x 3 unitary rotation matrix (or stack of matrices).
    Returns
    -------
    angle : `Angle`
        The angle of rotation.
    axis : array
        The (normalized) axis of rotation (with last dimension 3).
    """
    m = np.asanyarray(matrix)
    if m.shape[-2:] != (3, 3):
        raise ValueError('matrix is not 3x3')

    axis = np.zeros(m.shape[:-1])
    axis[..., 0] = m[..., 2, 1] - m[..., 1, 2]
    axis[..., 1] = m[..., 0, 2] - m[..., 2, 0]
    axis[..., 2] = m[..., 1, 0] - m[..., 0, 1]
    r = np.sqrt((axis * axis).sum(-1, keepdims=True))
    angle = np.arctan2(r[..., 0],
                       m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2] - 1.)
    return Angle(angle, u.radian), -axis / r
