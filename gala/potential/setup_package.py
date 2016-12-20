# from __future__ import absolute_import
# from distutils.core import Extension
# from astropy_helpers import setup_helpers

# def get_extensions():
#     exts = []

#     # malloc
#     mac_incl_path = "/usr/include/malloc"

#     # Potentials
#     cfg = setup_helpers.DistutilsExtensionArgs()
#     cfg['include_dirs'].append('numpy')
#     cfg['include_dirs'].append(mac_incl_path)
#     cfg['include_dirs'].append('gala/potential')
#     cfg['extra_compile_args'].append('--std=gnu99')
#     cfg['sources'].append('gala/potential/cpotential.pyx')
#     cfg['sources'].append('gala/potential/src/cpotential.c')
#     exts.append(Extension('gala.potential.cpotential', **cfg))

#     cfg = setup_helpers.DistutilsExtensionArgs()
#     cfg['include_dirs'].append('numpy')
#     cfg['include_dirs'].append(mac_incl_path)
#     cfg['include_dirs'].append('gala/potential')
#     cfg['extra_compile_args'].append('--std=gnu99')
#     cfg['sources'].append('gala/potential/ccompositepotential.pyx')
#     cfg['sources'].append('gala/potential/src/cpotential.c')
#     exts.append(Extension('gala.potential.ccompositepotential', **cfg))

#     cfg = setup_helpers.DistutilsExtensionArgs()
#     cfg['include_dirs'].append('numpy')
#     cfg['include_dirs'].append(mac_incl_path)
#     cfg['include_dirs'].append('gala/potential')
#     cfg['extra_compile_args'].append('--std=gnu99')
#     cfg['sources'].append('gala/potential/builtin/cybuiltin.pyx')
#     cfg['sources'].append('gala/potential/builtin/src/builtin_potentials.c')
#     cfg['sources'].append('gala/potential/src/cpotential.c')
#     exts.append(Extension('gala.potential.builtin.cybuiltin', **cfg))

#     # Frames
#     cfg = setup_helpers.DistutilsExtensionArgs()
#     cfg['include_dirs'].append('numpy')
#     cfg['include_dirs'].append(mac_incl_path)
#     cfg['include_dirs'].append('gala/potential')
#     cfg['extra_compile_args'].append('--std=gnu99')
#     cfg['sources'].append('gala/potential/cframe.pyx')
#     exts.append(Extension('gala.potential.cframe', **cfg))

#     cfg = setup_helpers.DistutilsExtensionArgs()
#     cfg['include_dirs'].append('numpy')
#     cfg['include_dirs'].append(mac_incl_path)
#     cfg['include_dirs'].append('gala/potential')
#     cfg['extra_compile_args'].append('--std=gnu99')
#     cfg['sources'].append('gala/potential/builtin/frames.pyx')
#     cfg['sources'].append('gala/potential/builtin/src/builtin_frames.c')
#     exts.append(Extension('gala.potential.builtin.frames', **cfg))

#     return exts

def get_package_data():
    return {'gala.potential': ['src/funcdefs.h', 'potential/src/cpotential.h']}

