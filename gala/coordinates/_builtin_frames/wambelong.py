"""Astropy coordinate class for the Wambelong coordinate system """

__all__ = ["WambelongShipp19"]


import numpy as np
from astropy.coordinates import ICRS, StaticMatrixTransform, frame_transform_graph
from astropy.utils.decorators import format_doc
from .base import BaseStreamFrame, stream_doc


@format_doc(
    stream_doc,
    name="Wambelong",
    paper="Shipp et al. 2019",
    url="https://ui.adsabs.harvard.edu/abs/2019ApJ...885....3S",
)
class WambelongShipp19(BaseStreamFrame):
    """A coordinate system defined by the orbit of the Wambelong stream."""


R = np.array(
    [
        [0.07420259, 0.76149392, -0.6439107],
        [-0.64686868, -0.45466937, -0.61223907],
        [0.75898279, -0.46195539, -0.45884892],
    ]
)


@frame_transform_graph.transform(StaticMatrixTransform, ICRS, WambelongShipp19)
def icrs_to_wambelong():
    return R


@frame_transform_graph.transform(StaticMatrixTransform, WambelongShipp19, ICRS)
def wambelong_to_icrs():
    return R.T
