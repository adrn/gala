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
    cfg['include_dirs'].append('gala/potential')
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('gala/potential/frame/cframe.pyx')
    cfg['sources'].append('gala/potential/frame/src/cframe.c')
    exts.append(Extension('gala.potential.frame.cframe', **cfg))

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(mac_incl_path)
    cfg['include_dirs'].append('gala/potential')
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('gala/potential/frame/builtin/frames.pyx')
    cfg['sources'].append('gala/potential/frame/builtin/builtin_frames.c')
    cfg['sources'].append('gala/potential/frame/src/cframe.c')
    exts.append(Extension('gala.potential.frame.builtin.frames', **cfg))

    return exts

def get_package_data():
    return {'gala.potential.frame':
            ['*.h', '*.pyx', '*.pxd', '*/*.pyx', '*/*.pxd',
             'builtin/*.h', 'src/*.h',
             'builtin/builtin_frames.c', 'src/*.c',
             'tests/*.yml']}
