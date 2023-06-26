"""Astropy coordinate class for the Molonglo coordinate system """

__all__ = ["MolongloShipp19"]


import numpy as np
from astropy.coordinates import ICRS, StaticMatrixTransform, frame_transform_graph
from astropy.utils.decorators import format_doc
from .base import BaseStreamFrame, stream_doc


@format_doc(
    stream_doc,
    name="Molonglo",
    paper="Shipp et al. 2019",
    url="https://ui.adsabs.harvard.edu/abs/2019ApJ...885....3S",
)
class MolongloShipp19(BaseStreamFrame):
    """A coordinate system defined by the orbit of the Molonglo stream."""


R = np.array(
    [
        [0.88306113, 0.15479520, -0.44299152],
        [0.36694639, -0.81621072, 0.44626270],
        [0.29249510, 0.55663139, 0.77756550],
    ]
)


@frame_transform_graph.transform(StaticMatrixTransform, ICRS, MolongloShipp19)
def icrs_to_molonglo():
    return R


@frame_transform_graph.transform(StaticMatrixTransform, MolongloShipp19, ICRS)
def molonglo_to_icrs():
    return R.T
