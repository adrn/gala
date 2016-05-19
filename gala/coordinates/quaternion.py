# coding: utf-8

""" Quaternions. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np

__all__ = ['Quaternion']

class Quaternion(object):
    """
    Quaternions for 3D rotations.

    This is largely thanks to Jake Vanderplas:
    https://jakevdp.github.io/blog/2012/11/24/simple-3d-visualization-in-matplotlib/

    Parameters
    ----------
    wxyz : array_like
        A quaternion in vector representation. Order is assumed
        to be ``(w,x,y,z)``.
    """
    def __init__(self, wxyz):
        self.wxyz = np.array(wxyz).astype(float)
        if self.wxyz.ndim > 1:
            raise NotImplementedError("Doesn't yet support array quaternions.")

    @classmethod
    def from_v_theta(cls, v, theta):
        """
        Create a quaternion from unit vector v and rotation angle theta.

        Returns
        -------
        q : :class:`gala.coordinates.Quaternion`
            A ``Quaternion`` instance.

        """
        theta = np.asarray(theta)
        v = np.asarray(v)

        s = np.sin(0.5 * theta)
        c = np.cos(0.5 * theta)
        vnrm = np.sqrt(np.sum(v * v))

        q = np.concatenate([[c], s * v / vnrm])
        return cls(q)

    def __repr__(self):
        return "Quaternion:\n" + self.wxyz.__repr__()

    def __str__(self):
        return "Quaternion({:.5f},{:.5f},{:.5f},{:.5f})".format(*self.wxyz)

    def __mul__(self, other):
        # multiplication of two quaternions.
        prod = self.wxyz[:, None] * other.wxyz
        return self.__class__([(prod[0, 0] - prod[1, 1]
                                 - prod[2, 2] - prod[3, 3]),
                                (prod[0, 1] + prod[1, 0]
                                 + prod[2, 3] - prod[3, 2]),
                                (prod[0, 2] - prod[1, 3]
                                 + prod[2, 0] + prod[3, 1]),
                                (prod[0, 3] + prod[1, 2]
                                 - prod[2, 1] + prod[3, 0])])

    @property
    def v_theta(self):
        """
        Return the ``(v, theta)`` equivalent of the (normalized) quaternion.

        Returns
        -------
        v : float
        theta : float

        """
        # compute theta
        norm = np.sqrt(np.sum(self.wxyz**2))
        theta = 2 * np.arccos(self.wxyz[0] / norm)

        # compute the unit vector
        v = np.array(self.wxyz[1:])
        v = v / np.sqrt(np.sum(v**2))

        return v, theta

    @property
    def rotation_matrix(self):
        """
        Compute the rotation matrix of the (normalized) quaternion.

        Returns
        -------
        R : :class:`~numpy.ndarray`
            A 3 by 3 rotation matrix (has shape ``(3,3)``).

        """
        v, theta = self.v_theta
        c = np.cos(theta)
        s = np.sin(theta)

        return np.array([[v[0] * v[0] * (1. - c) + c,
                          v[0] * v[1] * (1. - c) - v[2] * s,
                          v[0] * v[2] * (1. - c) + v[1] * s],
                         [v[1] * v[0] * (1. - c) + v[2] * s,
                          v[1] * v[1] * (1. - c) + c,
                          v[1] * v[2] * (1. - c) - v[0] * s],
                         [v[2] * v[0] * (1. - c) - v[1] * s,
                          v[2] * v[1] * (1. - c) + v[0] * s,
                          v[2] * v[2] * (1. - c) + c]])

    @classmethod
    def random(cls):
        """
        Randomly sample a Quaternion from a distribution uniform in
        3D rotation angles.

        https://www-preview.ri.cmu.edu/pub_files/pub4/kuffner_james_2004_1/kuffner_james_2004_1.pdf

        Returns
        -------
        q : :class:`gala.coordinates.Quaternion`
            A randomly sampled ``Quaternion`` instance.

        """

        s = np.random.uniform()
        s1 = np.sqrt(1 - s)
        s2 = np.sqrt(s)
        t1 = np.random.uniform(0, 2*np.pi)
        t2 = np.random.uniform(0, 2*np.pi)

        w = np.cos(t2)*s2
        x = np.sin(t1)*s1
        y = np.cos(t1)*s1
        z = np.sin(t2)*s2

        return cls([w,x,y,z])
