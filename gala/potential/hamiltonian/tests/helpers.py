# Third-party
import astropy.units as u
import numpy as np

# Project
from ....dynamics import PhaseSpacePosition, Orbit
from ....units import galactic
PSP = PhaseSpacePosition
ORB = Orbit


class _TestBase(object):
    use_half_ndim = False
    E_unit = u.erg/u.kg

    @classmethod
    def setup_class(cls):
        np.random.seed(42)

        ndim = 6
        r_ndim = ndim  # return ndim
        if cls.use_half_ndim:
            r_ndim = r_ndim // 2
        norbits = 16
        ntimes = 8

        # some position or phase-space position arrays we will test methods on:
        cls.w0s = []
        cls.energy_return_shapes = []
        cls.gradient_return_shapes = []
        cls.hessian_return_shapes = []

        # 1D - phase-space position
        cls.w0s.append(PSP(pos=np.random.random(size=ndim//2),
                           vel=np.random.random(size=ndim//2)))
        cls.w0s.append(PSP(pos=np.random.random(size=ndim//2)*u.kpc,
                           vel=np.random.random(size=ndim//2)*u.km/u.s))
        cls.energy_return_shapes += [(1,)]*2
        cls.gradient_return_shapes += [(r_ndim, 1)]*2
        cls.hessian_return_shapes += [(r_ndim, r_ndim, 1)]*2

        # 2D - phase-space position
        cls.w0s.append(PSP(pos=np.random.random(size=(ndim//2, norbits)),
                           vel=np.random.random(size=(ndim//2, norbits))))
        cls.w0s.append(PSP(pos=np.random.random(size=(ndim//2, norbits))*u.kpc,
                           vel=np.random.random(size=(ndim//2, norbits))*u.km/u.s))
        cls.energy_return_shapes += [(norbits,)]*2
        cls.gradient_return_shapes += [(r_ndim, norbits)]*2
        cls.hessian_return_shapes += [(r_ndim, r_ndim, norbits)]*2

        # 3D - phase-space position
        cls.w0s.append(PSP(pos=np.random.random(size=(ndim//2, norbits, ntimes)),
                           vel=np.random.random(size=(ndim//2, norbits, ntimes))))
        cls.w0s.append(PSP(pos=np.random.random(size=(ndim//2, norbits, ntimes))*u.kpc,
                           vel=np.random.random(size=(ndim//2, norbits, ntimes))*u.km/u.s))
        cls.energy_return_shapes += [(norbits, ntimes)]*2
        cls.gradient_return_shapes += [(r_ndim, norbits, ntimes)]*2
        cls.hessian_return_shapes += [(r_ndim, r_ndim, norbits, ntimes)]*2

        # 2D - orbit
        cls.w0s.append(ORB(pos=np.random.random(size=(ndim//2, ntimes)),
                           vel=np.random.random(size=(ndim//2, ntimes))))
        cls.w0s.append(ORB(pos=np.random.random(size=(ndim//2, ntimes))*u.kpc,
                           vel=np.random.random(size=(ndim//2, ntimes))*u.km/u.s))
        cls.energy_return_shapes += [(ntimes,)]*2
        cls.gradient_return_shapes += [(r_ndim, ntimes,)]*2
        cls.hessian_return_shapes += [(r_ndim, r_ndim, ntimes,)]*2

        # 3D - orbit
        cls.w0s.append(ORB(pos=np.random.random(size=(ndim//2, ntimes, norbits)),
                           vel=np.random.random(size=(ndim//2, ntimes, norbits))))
        cls.w0s.append(ORB(pos=np.random.random(size=(ndim//2, ntimes, norbits))*u.kpc,
                           vel=np.random.random(size=(ndim//2, ntimes, norbits))*u.km/u.s))
        cls.energy_return_shapes += [(ntimes, norbits)]*2
        cls.gradient_return_shapes += [(r_ndim, ntimes, norbits)]*2
        cls.hessian_return_shapes += [(r_ndim, r_ndim, ntimes, norbits)]*2

        _obj_w0s = cls.w0s[:]
        for w0, eshp, gshp, hshp in zip(_obj_w0s,
                                        cls.energy_return_shapes,
                                        cls.gradient_return_shapes,
                                        cls.hessian_return_shapes):
            cls.w0s.append(w0.w(galactic))
            cls.energy_return_shapes.append(eshp)
            cls.gradient_return_shapes.append(gshp)
            cls.hessian_return_shapes.append(hshp)

    def test_energy(self):
        for arr, shp in zip(self.w0s, self.energy_return_shapes):
            if self.E_unit.is_equivalent(u.one) and hasattr(arr, 'pos') and \
                    not arr.xyz.unit.is_equivalent(u.one):
                continue

            v = self.obj.energy(arr)
            assert v.shape == shp
            assert v.unit.is_equivalent(self.E_unit)

            t = np.zeros(np.array(arr).shape[1:]) + 0.1
            self.obj.energy(arr, t=0.1)
            self.obj.energy(arr, t=t)
            self.obj.energy(arr, t=0.1*self.obj.units['time'])

    def test_gradient(self):
        for arr, shp in zip(self.w0s, self.gradient_return_shapes):
            if self.E_unit.is_equivalent(u.one) and hasattr(arr, 'pos') and \
                    not arr.xyz.unit.is_equivalent(u.one):
                continue

            v = self.obj.gradient(arr)
            assert v.shape == shp
            # TODO: check return units

            t = np.zeros(np.array(arr).shape[1:]) + 0.1
            self.obj.gradient(arr, t=0.1)
            self.obj.gradient(arr, t=t)
            self.obj.gradient(arr, t=0.1*self.obj.units['time'])

    def test_hessian(self):
        for arr, shp in zip(self.w0s, self.hessian_return_shapes):
            if self.E_unit.is_equivalent(u.one) and hasattr(arr, 'pos') and \
                    not arr.xyz.unit.is_equivalent(u.one):
                continue

            g = self.obj.hessian(arr)
            assert g.shape == shp
            # TODO: check return units
