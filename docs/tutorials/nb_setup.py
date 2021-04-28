# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +

get_ipython().magic('config InlineBackend.figure_format = "retina"')  # noqa

import pathlib
import sys
import matplotlib.pyplot as plt

plt.style.use("default")

docs_root = (pathlib.Path(__file__).parent / '..').resolve()
sys.path.append(str(docs_root))
from conf import plot_rcparams

plt.rcParams.update(plot_rcparams)
