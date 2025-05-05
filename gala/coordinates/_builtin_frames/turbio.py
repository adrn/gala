"""Astropy coordinate class for the Turbio coordinate system """

__all__ = ["TurbioShipp19"]


import numpy as np
from astropy.coordinates import ICRS, StaticMatrixTransform, frame_transform_graph
from astropy.utils.decorators import format_doc
from .base import BaseStreamFrame, stream_doc


@format_doc(
    stream_doc,
    name="Turbio",
    paper="Shipp et al. 2019",
    url="https://ui.adsabs.harvard.edu/abs/2019ApJ...885....3S",
)
class TurbioShipp19(BaseStreamFrame):
    """A coordinate system defined by the orbit of the Turbio stream."""


R = np.array(
    [
        [0.52548400, 0.27871230, -0.80385697],
        [-0.71193491, -0.37328255, -0.59481831],
        [0.46584896, -0.88486134, 0.00227102],
    ]
)


@frame_transform_graph.transform(StaticMatrixTransform, ICRS, TurbioShipp19)
def icrs_to_turbio():
    return R


@frame_transform_graph.transform(StaticMatrixTransform, TurbioShipp19, ICRS)
def turbio_to_icrs():
    return R.T
