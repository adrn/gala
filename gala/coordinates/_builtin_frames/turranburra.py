"""Astropy coordinate class for the Turranburra coordinate system """

__all__ = ["TurranburraShipp19"]


import numpy as np
from astropy.coordinates import ICRS, StaticMatrixTransform, frame_transform_graph
from astropy.utils.decorators import format_doc
from .base import BaseStreamFrame, stream_doc


@format_doc(
    stream_doc,
    name="Turranburra",
    paper="Shipp et al. 2019",
    url="https://ui.adsabs.harvard.edu/abs/2019ApJ...885....3S",
)
class TurranburraShipp19(BaseStreamFrame):
    """A coordinate system defined by the orbit of the Turranburra stream."""


R = np.array(
    [
        [0.36111266, 0.85114984, -0.38097455],
        [0.87227667, -0.16384562, 0.46074725],
        [-0.32974393, 0.49869687, 0.80160487],
    ]
)


@frame_transform_graph.transform(StaticMatrixTransform, ICRS, TurranburraShipp19)
def icrs_to_turranburra():
    return R


@frame_transform_graph.transform(StaticMatrixTransform, TurranburraShipp19, ICRS)
def turranburra_to_icrs():
    return R.T
