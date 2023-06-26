"""Astropy coordinate class for the Indus coordinate system """

__all__ = ["IndusShipp19"]

import numpy as np
from astropy.coordinates import ICRS, StaticMatrixTransform, frame_transform_graph
from astropy.utils.decorators import format_doc
from .base import BaseStreamFrame, stream_doc

@format_doc(
    stream_doc,
    name="Indus",
    paper="Shipp et al. 2019",
    url="https://ui.adsabs.harvard.edu/abs/2019ApJ...885....3S",
)
class IndusShipp19(BaseStreamFrame):
    """A coordinate system defined by the orbit of the Indus stream."""


R = np.array(
    [
        [0.47348784, -0.22057954, -0.85273321],
        [0.25151201, -0.89396596, 0.37089969],
        [0.84412734, 0.39008914, 0.36780360],
    ]
)


@frame_transform_graph.transform(StaticMatrixTransform, ICRS, IndusShipp19)
def icrs_to_indus():
    return R


@frame_transform_graph.transform(StaticMatrixTransform, IndusShipp19, ICRS)
def indus_to_icrs():
    return R.T
