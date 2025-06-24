import pickle

import astropy.units as u
import pytest

from ....units import DimensionlessUnitSystem, galactic
from ..builtin import ConstantRotatingFrame, StaticFrame


class TestStaticFrame:
    def test_init(self):
        fr = StaticFrame()
        assert isinstance(fr.units, DimensionlessUnitSystem)

        fr = StaticFrame(galactic)

    def test_compare(self):
        fr1 = StaticFrame(galactic)
        fr2 = StaticFrame(galactic)
        assert fr1 == fr2

        fr2 = StaticFrame()
        assert fr1 != fr2

    def test_pickle(self, tmpdir):
        fr1 = StaticFrame(galactic)

        filename = tmpdir / "static.pkl"
        with open(filename, "wb") as f:
            pickle.dump(fr1, f)

        with open(filename, "rb") as f:
            fr2 = pickle.load(f)

        assert fr1 == fr2


class TestConstantRotatingFrame:
    def test_init(self):
        fr = ConstantRotatingFrame(Omega=[1e-3, 0.0, 0.0])
        assert isinstance(fr.units, DimensionlessUnitSystem)

        fr = ConstantRotatingFrame(Omega=1e-3)
        assert isinstance(fr.units, DimensionlessUnitSystem)

        with pytest.raises(ValueError):
            fr = ConstantRotatingFrame(Omega=[-13.0, 1.0, 40.0] * u.km / u.s / u.kpc)

        with pytest.raises(ValueError):
            fr = ConstantRotatingFrame(Omega=40.0 * u.km / u.s / u.kpc)

        fr = ConstantRotatingFrame(
            Omega=[-13.0, 1.0, 40.0] * u.km / u.s / u.kpc, units=galactic
        )
        fr = ConstantRotatingFrame(Omega=40.0 * u.km / u.s / u.kpc, units=galactic)
        fr = ConstantRotatingFrame(
            [-13.0, 1.0, 40.0] * u.km / u.s / u.kpc, units=galactic
        )

    def test_compare(self):
        # frame comparison
        fr1 = ConstantRotatingFrame(Omega=[1e-3, 0.0, 0.0] / u.Myr, units=galactic)
        fr2 = ConstantRotatingFrame(Omega=[1e-3, 0.0, 0.0] / u.Myr, units=galactic)
        fr3 = ConstantRotatingFrame(Omega=[2e-3, 0.0, 0.0] / u.Myr, units=galactic)
        fr4 = ConstantRotatingFrame(Omega=[2e-3, 0.0, 0.0])
        assert fr1 == fr2
        assert fr1 != fr3
        assert fr3 != fr4

        st_fr = StaticFrame(galactic)
        assert st_fr != fr1

        st_fr = StaticFrame(DimensionlessUnitSystem())
        assert st_fr != fr1

    def test_pickle(self, tmpdir):
        fr1 = ConstantRotatingFrame(Omega=[1e-3, 0.0, 0.0] / u.Myr, units=galactic)

        filename = tmpdir / "rotating.pkl"
        with open(filename, "wb") as f:
            pickle.dump(fr1, f)

        with open(filename, "rb") as f:
            fr2 = pickle.load(f)

        assert fr1 == fr2
