from distutils.core import Extension
from collections import defaultdict


def get_extensions():
    import numpy as np

    exts = []

    # malloc
    mac_incl_path = "/usr/include/malloc"

    cfg = defaultdict(list)
    cfg['include_dirs'].append(np.get_include())
    cfg['include_dirs'].append(mac_incl_path)
    cfg['include_dirs'].append('gala/integrate/cyintegrators')
    cfg['include_dirs'].append('gala/potential')
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('gala/potential/potential/src/cpotential.c')
    cfg['sources'].append('gala/potential/hamiltonian/src/chamiltonian.c')
    cfg['sources'].append('gala/integrate/cyintegrators/dopri/dop853.c')
    cfg['sources'].append('gala/dynamics/lyapunov/dop853_lyapunov.pyx')
    exts.append(Extension('gala.dynamics.lyapunov.dop853_lyapunov', **cfg))

    cfg = defaultdict(list)
    cfg['include_dirs'].append(np.get_include())
    cfg['include_dirs'].append(mac_incl_path)
    cfg['include_dirs'].append('gala/potential')
    cfg['sources'].append('gala/dynamics/mockstream/_coord.pyx')
    cfg['extra_compile_args'].append('--std=gnu99')
    exts.append(Extension('gala.dynamics.mockstream._coord', **cfg))

    cfg = defaultdict(list)
    cfg['include_dirs'].append(np.get_include())
    cfg['include_dirs'].append(mac_incl_path)
    cfg['include_dirs'].append('gala/potential')
    cfg['sources'].append('gala/dynamics/mockstream/df.pyx')
    cfg['sources'].append('gala/potential/potential/src/cpotential.c')
    cfg['extra_compile_args'].append('--std=gnu99')
    exts.append(Extension('gala.dynamics.mockstream.df', **cfg))

    cfg = defaultdict(list)
    cfg['include_dirs'].append(np.get_include())
    cfg['include_dirs'].append('gala/integrate/cyintegrators')
    cfg['include_dirs'].append(mac_incl_path)
    cfg['include_dirs'].append('gala/potential')
    cfg['include_dirs'].append('gala/dynamics/nbody')
    cfg['sources'].append('gala/potential/potential/src/cpotential.c')
    cfg['sources'].append('gala/potential/hamiltonian/src/chamiltonian.c')
    cfg['sources'].append('gala/dynamics/mockstream/mockstream.pyx')
    cfg['sources'].append('gala/integrate/cyintegrators/dopri/dop853.c')
    cfg['extra_compile_args'].append('--std=gnu99')
    exts.append(Extension('gala.dynamics.mockstream._mockstream', **cfg))

    cfg = defaultdict(list)
    cfg['include_dirs'].append(np.get_include())
    cfg['include_dirs'].append('gala/integrate/cyintegrators')
    cfg['include_dirs'].append(mac_incl_path)
    cfg['include_dirs'].append('gala/potential')
    cfg['sources'].append('gala/potential/potential/src/cpotential.c')
    cfg['sources'].append('gala/potential/hamiltonian/src/chamiltonian.c')
    cfg['sources'].append('gala/integrate/cyintegrators/dopri/dop853.c')
    cfg['sources'].append('gala/dynamics/nbody/nbody.pyx')
    cfg['extra_compile_args'].append('--std=gnu99')
    exts.append(Extension('gala.dynamics.nbody.nbody', **cfg))

    return exts
