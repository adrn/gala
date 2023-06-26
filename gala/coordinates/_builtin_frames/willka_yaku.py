"""Astropy coordinate class for the WillkaYaku coordinate system """

__all__ = ["WillkaYakuShipp19"]


import numpy as np
from astropy.coordinates import ICRS, StaticMatrixTransform, frame_transform_graph
from astropy.utils.decorators import format_doc
from .base import BaseStreamFrame, stream_doc


@format_doc(
    stream_doc,
    name="Willka-Yaku",
    paper="Shipp et al. 2019",
    url="https://ui.adsabs.harvard.edu/abs/2019ApJ...885....3S",
)
class WillkaYakuShipp19(BaseStreamFrame):
    """A coordinate system defined by the orbit of the Willka-Yaku stream."""


R = np.array(
    [
        [0.37978305, 0.29001265, -0.87844038],
        [-0.5848418, -0.66046543, -0.47089859],
        [0.71674605, -0.69258795, -0.08122206],
    ]
)


@frame_transform_graph.transform(StaticMatrixTransform, ICRS, WillkaYakuShipp19)
def icrs_to_willkayaku():
    return R


@frame_transform_graph.transform(StaticMatrixTransform, WillkaYakuShipp19, ICRS)
def willkayaku_to_icrs():
    return R.T
