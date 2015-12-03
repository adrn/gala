from __future__ import absolute_import
from distutils.core import Extension
from astropy_helpers import setup_helpers

def get_extensions():
    exts = []

    # malloc
    mac_incl_path = "/usr/include/malloc"

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(mac_incl_path)
    cfg['include_dirs'].append('gary/integrate/cyintegrators/')
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('gary/integrate/cyintegrators/dopri/dop853.c')
    cfg['sources'].append('gary/dynamics/lyapunov/dop853_lyapunov.pyx')
    exts.append(Extension('gary.dynamics.lyapunov.dop853_lyapunov', **cfg))

    return exts
