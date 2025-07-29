"""General utilities."""

import copy
import os
from collections.abc import Mapping
from contextlib import contextmanager

import numpy as np

__all__ = ["assert_angles_allclose", "atleast_2d", "rolling_window"]


class ImmutableDict(Mapping):
    @classmethod
    def from_dict(cls, somedict):
        return cls(**somedict)

    def __init__(self, **kwargs):
        self._dict = kwargs
        self._hash = None

    def __getitem__(self, key):
        return self._dict[key]

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(frozenset(self._dict.items()))
        return self._hash

    def __eq__(self, other):
        return self._dict == other._dict

    def __repr__(self):
        return f"<ImmutableDict {self._dict.__repr__()}>"

    def __str__(self):
        return self._dict.__str__()

    def copy(self):
        return copy.deepcopy(self._dict)


def rolling_window(arr, window_size, stride=1, return_idx=False):
    """
    There is an example of an iterator for pure-Python objects in:
    http://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator-in-python
    This is a rolling-window iterator Numpy arrays, with window size and
    stride control. See examples below for demos.

    Parameters
    ----------
    arr : array_like
        Input numpy array.
    window_size : int
        Width of the window.
    stride : int (optional)
        Number of indices to advance the window each iteration step.
    return_idx : bool (optional)
        Whether to return the slice indices alone with the array segment.

    Examples
    --------
    >>> a = np.array([1, 2, 3, 4, 5, 6])
    >>> for x in rolling_window(a, 3):
    ...     print(x)
    [1 2 3]
    [2 3 4]
    [3 4 5]
    [4 5 6]
    >>> for x in rolling_window(a, 2, stride=2):
    ...     print(x)
    [1 2]
    [3 4]
    [5 6]
    >>> for (i1, i2), x in rolling_window(a, 2, stride=2, return_idx=True): # doctest: +SKIP
    ...     print(i1, i2, x)
    (0, 2, array([1, 2]))
    (2, 4, array([3, 4]))
    (4, 6, array([5, 6]))

    """

    window_size = int(window_size)
    stride = int(stride)

    if window_size < 0 or stride < 1:
        raise ValueError

    arr_len = len(arr)
    if arr_len < window_size:
        if return_idx:
            yield (0, arr_len), arr
        else:
            yield arr

    ix1 = 0
    while ix1 < arr_len:
        ix2 = ix1 + window_size
        result = arr[ix1:ix2]
        if return_idx:
            yield (ix1, ix2), result
        else:
            yield result
        if len(result) < window_size or ix2 >= arr_len:
            break
        ix1 += stride


def atleast_2d(*arys, **kwargs):
    """
    View inputs as arrays with at least two dimensions.

    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more array-like sequences.  Non-array inputs are converted
        to arrays.  Arrays that already have two or more dimensions are
        preserved.
    insert_axis : int (optional)
        Where to create a new axis if input array(s) have <2 dim.

    Returns
    -------
    res, res2, ... : ndarray
        An array, or tuple of arrays, each with ``a.ndim >= 2``.
        Copies are avoided where possible, and views with two or more
        dimensions are returned.

    Examples
    --------
    >>> atleast_2d(3.0) # doctest: +FLOAT_CMP
    array([[3.]])

    >>> x = np.arange(3.0)
    >>> atleast_2d(x) # doctest: +FLOAT_CMP
    array([[0., 1., 2.]])
    >>> atleast_2d(x, insert_axis=-1) # doctest: +FLOAT_CMP
    array([[0.],
           [1.],
           [2.]])
    >>> atleast_2d(x).base is x
    True

    >>> atleast_2d(1, [1, 2], [[1, 2]])
    [array([[1]]), array([[1, 2]]), array([[1, 2]])]

    """
    insert_axis = kwargs.pop("insert_axis", 0)
    slc = [slice(None)] * 2
    slc[insert_axis] = None
    slc = tuple(slc)

    res = []
    for ary in arys:
        ary = np.asanyarray(ary)
        if len(ary.shape) == 0:
            result = ary.reshape(1, 1)
        elif len(ary.shape) == 1:
            result = ary[slc]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    return res


def assert_angles_allclose(x, y, **kwargs):
    """
    Like numpy's assert_allclose, but for angles (in radians).
    """
    c2 = (np.sin(x) - np.sin(y)) ** 2 + (np.cos(x) - np.cos(y)) ** 2
    diff = np.arccos((2.0 - c2) / 2.0)  # a = b = 1
    assert np.allclose(diff, 0.0, **kwargs)


class GalaDeprecationWarning(DeprecationWarning):
    """
    A warning class to indicate a deprecated feature.
    """


@contextmanager
def chdir(new_path):
    """
    Context manager to change the current working directory.

    Parameters
    ----------
    new_path : str
        The path to change to.
    """
    old_path = os.getcwd()
    os.chdir(new_path)
    try:
        yield
    finally:
        os.chdir(old_path)
