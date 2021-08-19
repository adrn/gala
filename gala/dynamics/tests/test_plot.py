""" Test dynamics plotting functions """

# Third-party
import numpy as np
import astropy.units as u
import pytest

# Project
from ..core import PhaseSpacePosition
from ..orbit import Orbit
from ..plot import plot_projections


def pytest_generate_tests(metafunc):
    if "obj" not in metafunc.fixturenames:
        return

    object_list = []

    norbits = 16
    object_list.append(
        PhaseSpacePosition(pos=np.random.random(size=3),
                           vel=np.random.random(size=3)))
    object_list.append(
        PhaseSpacePosition(pos=np.random.random(size=(3, norbits)),
                           vel=np.random.random(size=(3, norbits))))
    object_list.append(
        PhaseSpacePosition(pos=np.random.random(size=(3, norbits))*u.kpc,
                           vel=np.random.random(size=(3, norbits))*u.km/u.s))

    nsteps = 16
    object_list.append(Orbit(pos=np.random.random(size=(3, nsteps)),
                             vel=np.random.random(size=(3, nsteps)),
                             t=np.linspace(0, 1, nsteps)))
    object_list.append(Orbit(pos=np.random.random(size=(3, nsteps, 2)),
                             vel=np.random.random(size=(3, nsteps, 2)),
                             t=np.linspace(0, 1, nsteps)))
    object_list.append(Orbit(pos=np.random.random(size=(3, nsteps))*u.kpc,
                             vel=np.random.random(size=(3, nsteps))*u.km/u.s,
                             t=np.linspace(0, 1, nsteps)*u.Myr))

    # 2D
    object_list.append(
        PhaseSpacePosition(pos=np.random.random(size=(2, norbits)),
                           vel=np.random.random(size=(2, norbits))))
    object_list.append(Orbit(pos=np.random.random(size=(2, nsteps)),
                             vel=np.random.random(size=(2, nsteps)),
                             t=np.linspace(0, 1, nsteps)))

    test_names = [f'{obj.__class__.__name__}{i}'
                  for i, obj in enumerate(object_list)]

    metafunc.parametrize(['i', 'obj'],
                         list(enumerate(object_list)),
                         ids=test_names)


def test_plot_projections(i, obj):
    import matplotlib.pyplot as plt

    # Try executing the method - unfortunately no test of the actual figure
    # drawn!
    obj.plot()

    # Try with just 2D projection, and passing in a bunch of inputs...
    x = obj.xyz.value
    fig, axes = plt.subplots(1, 2)
    fig = plot_projections(x[:2], autolim=True, axes=axes,  # noqa
                           subplots_kwargs=dict(sharex=True),
                           labels=['x', 'y'],
                           plot_function=plt.plot,
                           marker='o', linestyle='--', color='r')


def test_animate(tmpdir, i, obj):
    if not isinstance(obj, Orbit):
        pytest.skip()

    # Try executing the method - unfortunately no test of the actual figure
    # drawn!
    fig, anim = obj.animate(segment_nsteps=3)
    anim.save(tmpdir / f'anim{i}.mp4')

    if obj.ndim == 3:
        # Also try cylindrical, and sub-selecting components:
        fig, anim = obj.cylindrical.animate(components=['rho', 'z'])
        anim.save(tmpdir / f'anim{i}_cyl.mp4')
