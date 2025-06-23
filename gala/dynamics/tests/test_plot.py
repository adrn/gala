"""Test dynamics plotting functions"""

import subprocess

import astropy.units as u
import numpy as np
import pytest

from gala.tests.optional_deps import HAS_MATPLOTLIB
from gala.units import galactic

from ..core import PhaseSpacePosition
from ..orbit import Orbit
from ..plot import plot_projections

if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt
else:
    plt = None


def pytest_generate_tests(metafunc):
    if "obj" not in metafunc.fixturenames:
        return

    object_list = []

    norbits = 16
    object_list.append(
        PhaseSpacePosition(pos=np.random.random(size=3), vel=np.random.random(size=3))
    )
    object_list.append(
        PhaseSpacePosition(
            pos=np.random.random(size=(3, norbits)),
            vel=np.random.random(size=(3, norbits)),
        )
    )
    object_list.append(
        PhaseSpacePosition(
            pos=np.random.random(size=(3, norbits)) * u.kpc,
            vel=np.random.random(size=(3, norbits)) * u.km / u.s,
        )
    )

    nsteps = 16
    object_list.append(
        Orbit(
            pos=np.random.random(size=(3, nsteps)),
            vel=np.random.random(size=(3, nsteps)),
            t=np.linspace(0, 1, nsteps),
        )
    )
    object_list.append(
        Orbit(
            pos=np.random.random(size=(3, nsteps, 2)),
            vel=np.random.random(size=(3, nsteps, 2)),
            t=np.linspace(0, 1, nsteps),
        )
    )
    object_list.append(
        Orbit(
            pos=np.random.random(size=(3, nsteps)) * u.kpc,
            vel=np.random.random(size=(3, nsteps)) * u.km / u.s,
            t=np.linspace(0, 1, nsteps) * u.Myr,
        )
    )

    # 2D
    object_list.append(
        PhaseSpacePosition(
            pos=np.random.random(size=(2, norbits)),
            vel=np.random.random(size=(2, norbits)),
        )
    )
    object_list.append(
        Orbit(
            pos=np.random.random(size=(2, nsteps)),
            vel=np.random.random(size=(2, nsteps)),
            t=np.linspace(0, 1, nsteps),
        )
    )

    test_names = [f"{obj.__class__.__name__}{i}" for i, obj in enumerate(object_list)]

    metafunc.parametrize(["i", "obj"], list(enumerate(object_list)), ids=test_names)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Matplotlib is required")
def test_plot_projections(i, obj):
    # Try executing the method
    # TODO: no test of the actual figure drawn!
    fig = obj.plot()
    plt.close(fig)

    # Try with just 2D projection, and passing in a bunch of inputs...
    x = obj.xyz.value
    fig, axes = plt.subplots(1, 2)
    fig = plot_projections(
        x[:2],
        autolim=True,
        axes=axes,
        subplots_kwargs={"sharex": True},
        labels=["x", "y"],
        plot_function=plt.plot,
        marker="o",
        linestyle="--",
        color="r",
    )
    plt.close(fig)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Matplotlib is required")
def test_units(i, obj):
    comp_names = list(obj.pos_components.keys())

    if getattr(obj, comp_names[0]).unit == u.one:
        with pytest.raises(u.UnitConversionError):
            obj.plot(comp_names[:2], units=u.kpc)

        with pytest.raises(u.UnitConversionError):
            obj.plot(comp_names[:2], units=[u.kpc, u.pc])

        fig = obj.plot(units=galactic)
        plt.close(fig)

    else:
        fig = obj.plot(comp_names[:2], units=u.kpc)
        plt.close(fig)

        fig = obj.plot(comp_names[:2], units=[u.kpc, u.pc])
        plt.close(fig)

        fig = obj.plot(comp_names[:2], units=galactic)
        plt.close(fig)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Matplotlib is required")
def test_animate(tmpdir, i, obj):
    if not isinstance(obj, Orbit):
        pytest.skip()

    try:
        proc = subprocess.run(
            ["ffmpeg -version"], shell=True, check=True, capture_output=True
        )
    except subprocess.CalledProcessError:
        pytest.skip(reason="ffmpeg not installed")

    if proc.returncode > 0:
        pytest.skip(reason="ffmpeg not installed")

    # Try executing the method - unfortunately no test of the actual figure
    # drawn!
    fig, anim = obj.animate(segment_nsteps=3)
    anim.save(tmpdir / f"anim{i}.mp4")

    # test hiding the timestep label
    fig, anim = obj.animate(segment_nsteps=3, show_time=False)
    anim.save(tmpdir / f"anim{i}_no_time.mp4")

    if obj.ndim == 3:
        # Also try cylindrical, and sub-selecting components:
        _fig, anim = obj.cylindrical.animate(components=["rho", "z"])
        anim.save(tmpdir / f"anim{i}_cyl.mp4")
