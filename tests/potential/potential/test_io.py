"""test reading/writing potentials to files"""

import io
from enum import Enum
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
import yaml
from gala._cconfig import GSL_ENABLED
from scipy.spatial.transform import Rotation

from gala.potential import (
    CCompositePotential,
    CompositePotential,
    HernquistPotential,
    IsochronePotential,
    KeplerPotential,
    LM10Potential,
    LongMuraliBarPotential,
    MilkyWayPotential,
    NFWPotential,
    NullPotential,
    SCFPotential,
)
from gala.potential.potential.io import (
    _pack_params,
    _to_yaml_safe,
    load,
    save,
    to_dict,
)
from gala.units import DimensionlessUnitSystem, UnitSystem, galactic

this_path = Path(__file__).parent


def test_read_plummer():
    potential = load(this_path / "Plummer.yml")
    assert np.allclose(potential.parameters["m"].value, 100000000000.0)
    assert np.allclose(potential.parameters["b"].value, 0.26)
    assert potential.parameters["b"].unit == u.kpc


def test_read_harmonic_oscillator():
    potential = load(this_path / "HarmonicOscillator1D.yml")
    assert isinstance(potential.units, DimensionlessUnitSystem)


def test_read_composite():
    potential = load(this_path / "Composite.yml")
    assert "halo" in potential
    assert "disk" in potential
    assert str(potential) == "CompositePotential"
    assert potential.units["length"] == u.kpc
    assert potential.units["speed"] == u.km / u.s


def test_read_lm10():
    potential = load(this_path / "lm10.yml")
    assert "halo" in potential
    assert "disk" in potential
    assert str(potential) == "LM10Potential"
    assert np.allclose(potential["disk"].parameters["a"].value, 10)
    assert np.allclose(potential["disk"].parameters["b"].value, 0.26)
    assert np.allclose(potential["disk"].parameters["m"].value, 150000.0)


def test_write_isochrone(tmpdir):
    tmp_filename = str(tmpdir.join("potential.yml"))

    # try a simple potential
    potential = IsochronePotential(m=1e11, b=0.76, units=galactic)

    with open(tmp_filename, "w", encoding="utf-8") as f:
        save(potential, f)

    save(potential, tmp_filename)
    p = load(tmp_filename)


def test_write_isochrone_units(tmpdir):
    tmp_filename = str(tmpdir.join("potential.yml"))

    # try a simple potential with units
    potential = IsochronePotential(m=1e11 * u.Msun, b=0.76 * u.kpc, units=galactic)

    with open(tmp_filename, "w", encoding="utf-8") as f:
        save(potential, f)

    save(potential, tmp_filename)
    p = load(tmp_filename)


def test_write_lm10(tmpdir):
    tmp_filename = str(tmpdir.join("potential.yml"))

    # more complex
    potential = LM10Potential(disk={"m": 5e12 * u.Msun})
    potential_default = LM10Potential()
    v1 = potential.energy([4.0, 0, 0])
    v2 = potential_default.energy([4.0, 0, 0])

    with open(tmp_filename, "w", encoding="utf-8") as f:
        save(potential, f)

    save(potential, tmp_filename)
    p = load(tmp_filename)
    assert u.allclose(p["disk"].parameters["m"], 5e12 * u.Msun)
    assert u.allclose(v1, p.energy([4.0, 0, 0]))
    assert not u.allclose(v2, p.energy([4.0, 0, 0]))


def test_write_composite(tmpdir):
    tmp_filename = str(tmpdir.join("potential.yml"))
    print(tmp_filename)

    # composite potential
    potential = CompositePotential(
        halo=KeplerPotential(m=1e11, units=galactic),
        bulge=IsochronePotential(m=1e11, b=0.76, units=galactic),
    )
    save(potential, tmp_filename)
    p = load(tmp_filename)


def test_write_ccomposite(tmpdir):
    tmp_filename = str(tmpdir.join("potential.yml"))

    # composite potential
    potential = CCompositePotential(
        halo=KeplerPotential(m=1e11, units=galactic),
        bulge=IsochronePotential(m=1e11, b=0.76, units=galactic),
    )
    save(potential, tmp_filename)
    p = load(tmp_filename)


def test_units(tmpdir):
    import astropy.units as u

    tmp_filename = str(tmpdir.join("potential.yml"))

    # try a simple potential
    potential = KeplerPotential(m=1e11, units=[u.kpc, u.Gyr, u.Msun, u.radian])
    save(potential, tmp_filename)
    p = load(tmp_filename)


@pytest.mark.skipif(not GSL_ENABLED, reason="requires GSL to run this test")
def test_read_write_SCF(tmpdir):
    tmp_filename = str(tmpdir.join("potential.yml"))

    # try a basic SCF potential
    potential = SCFPotential(100, 1, np.zeros((4, 3, 2)), np.zeros((4, 3, 2)))
    save(potential, tmp_filename)
    p = load(tmp_filename)


_sentinel_calls = []


def _sentinel_callable(*args, **kwargs):
    _sentinel_calls.append(("apply", args, kwargs))
    return {"class": "NullPotential"}


class _SentinelClass:
    def __init__(self, *args, **kwargs):
        _sentinel_calls.append(("new", args, kwargs))


@pytest.mark.parametrize(
    "payload_factory",
    [
        pytest.param(
            lambda mod: f"!!python/object/apply:{mod}._sentinel_callable []\n",
            id="object_apply_sentinel",
        ),
        pytest.param(
            lambda mod: f"!!python/object/new:{mod}._SentinelClass []\n",
            id="object_new_sentinel",
        ),
        pytest.param(
            lambda mod: f"!!python/name:{mod}._sentinel_callable ''\n",
            id="python_name_sentinel",
        ),
        pytest.param(
            lambda _mod: "!!python/object/apply:builtins.dict []\n",
            id="object_apply_builtins_dict",
        ),
    ],
)
def test_load_rejects_unsafe_yaml_file(tmp_path, payload_factory):
    mod = _sentinel_callable.__module__
    p = tmp_path / "unsafe.yml"
    p.write_text(payload_factory(mod), encoding="utf-8")
    _sentinel_calls.clear()
    with pytest.raises(yaml.constructor.ConstructorError):
        load(p)
    assert _sentinel_calls == []


def test_load_rejects_unsafe_yaml_file_like():
    mod = _sentinel_callable.__module__
    payload = f"!!python/object/apply:{mod}._sentinel_callable []\n"
    _sentinel_calls.clear()
    with pytest.raises(yaml.constructor.ConstructorError):
        load(io.StringIO(payload))
    assert _sentinel_calls == []


@pytest.mark.parametrize(
    ("potential", "check_energy"),
    [
        (MilkyWayPotential(version="v1"), True),
        (MilkyWayPotential(version="v2"), True),
        (
            CompositePotential(
                halo=NFWPotential(m=1e12 * u.Msun, r_s=15 * u.kpc, units=galactic),
                bulge=HernquistPotential(m=5e9 * u.Msun, c=1 * u.kpc, units=galactic),
            ),
            True,
        ),
        (
            CCompositePotential(
                halo=NFWPotential(m=1e12 * u.Msun, r_s=15 * u.kpc, units=galactic),
                bulge=HernquistPotential(m=5e9 * u.Msun, c=1 * u.kpc, units=galactic),
            ),
            True,
        ),
        (
            LongMuraliBarPotential(
                m=1e10 * u.Msun,
                a=4 * u.kpc,
                b=1 * u.kpc,
                c=1 * u.kpc,
                R=Rotation.from_euler("z", 35, degrees=True),
                units=galactic,
            ),
            False,
        ),
        (
            KeplerPotential(
                m=1e11 * u.Msun,
                units=UnitSystem(u.kpc, u.Gyr, u.Msun, u.rad),
            ),
            True,
        ),
    ],
)
def test_roundtrip_safe_yaml(tmp_path, potential, check_energy):
    fn = tmp_path / "potential.yml"
    save(potential, fn)
    raw = fn.read_text(encoding="utf-8")
    assert "!!python/" not in raw

    loaded_file = load(fn)
    loaded_buf = load(io.StringIO(raw))
    for loaded in (loaded_file, loaded_buf):
        assert loaded == potential
        if check_energy:
            pos = [4.0, 2.0, 1.0] * u.kpc
            assert u.allclose(loaded.energy(pos), potential.energy(pos))
            assert u.allclose(loaded.gradient(pos), potential.gradient(pos))


class _StrSubclass(str):
    pass


class _IntSubclass(int):
    pass


class _FloatSubclass(float):
    pass


def _assert_plain_yaml_types(obj):
    if obj is None or type(obj) in (str, int, float, bool, bytes):
        return
    if type(obj) is list:
        for item in obj:
            _assert_plain_yaml_types(item)
        return
    if type(obj) is dict:
        for k, v in obj.items():
            _assert_plain_yaml_types(k)
            _assert_plain_yaml_types(v)
        return
    pytest.fail(f"Non-plain YAML type leaked into to_dict(): {type(obj)!r} ({obj!r})")


@pytest.mark.parametrize("composite_cls", [CompositePotential, CCompositePotential])
@pytest.mark.parametrize(
    "label_factory",
    [
        pytest.param(lambda s: np.array([s])[0], id="np_str_"),
        pytest.param(_StrSubclass, id="str_subclass"),
    ],
)
def test_composite_str_subclass_component_names_roundtrip(
    tmp_path, composite_cls, label_factory
):
    halo_label = label_factory("halo")
    bulge_label = label_factory("bulge")
    assert isinstance(halo_label, str)
    assert type(halo_label) is not str

    potential = composite_cls()
    potential[halo_label] = NFWPotential(
        m=1e12 * u.Msun, r_s=15 * u.kpc, units=galactic
    )
    potential[bulge_label] = HernquistPotential(
        m=5e9 * u.Msun, c=1 * u.kpc, units=galactic
    )

    pot_dict = to_dict(potential)
    _assert_plain_yaml_types(pot_dict)
    assert [c["name"] for c in pot_dict["components"]] == ["halo", "bulge"]
    assert all(type(c["name"]) is str for c in pot_dict["components"])

    fn = tmp_path / "composite_np_str.yml"
    save(potential, fn)
    raw = fn.read_text(encoding="utf-8")
    assert "!!python/" not in raw

    buf = io.StringIO()
    save(potential, buf)
    assert buf.getvalue() == raw

    for loaded in (load(fn), load(io.StringIO(raw))):
        assert loaded == potential
        assert list(loaded.keys()) == ["halo", "bulge"]
        assert all(type(k) is str for k in loaded)
        pos = [4.0, 2.0, 1.0] * u.kpc
        assert u.allclose(loaded.energy(pos), potential.energy(pos))
        assert u.allclose(loaded.gradient(pos), potential.gradient(pos))


class _PartEnum(str, Enum):  # noqa: UP042
    HALO = "halo"


@pytest.mark.parametrize("composite_cls", [CompositePotential, CCompositePotential])
def test_composite_str_enum_component_name_no_collision(tmp_path, composite_cls):
    assert str(_PartEnum.HALO) == "_PartEnum.HALO"
    assert _PartEnum.HALO == "halo"
    assert hash(_PartEnum.HALO) == hash("halo")
    assert len({"halo": 1, _PartEnum.HALO: 2}) == 1

    potential = composite_cls()
    potential[_PartEnum.HALO] = KeplerPotential(m=3.0, units=DimensionlessUnitSystem())
    potential["_PartEnum.HALO"] = KeplerPotential(
        m=7.0, units=DimensionlessUnitSystem()
    )

    pos = [1.0, 0.0, 0.0]
    expected_energy = potential.energy(pos)
    assert np.allclose(expected_energy, [-10.0])

    pot_dict = to_dict(potential)
    _assert_plain_yaml_types(pot_dict)
    assert [c["name"] for c in pot_dict["components"]] == ["halo", "_PartEnum.HALO"]
    assert all(type(c["name"]) is str for c in pot_dict["components"])

    fn = tmp_path / "composite_str_enum.yml"
    save(potential, fn)
    raw = fn.read_text(encoding="utf-8")
    assert "!!python/" not in raw

    buf = io.StringIO()
    save(potential, buf)
    assert buf.getvalue() == raw

    for loaded in (load(fn), load(io.StringIO(raw))):
        assert loaded == potential
        assert list(loaded.keys()) == ["halo", "_PartEnum.HALO"]
        assert all(type(k) is str for k in loaded)
        assert np.allclose(loaded.energy(pos), [-10.0])
        assert np.allclose(loaded.energy(pos), expected_energy)
        assert np.allclose(loaded["halo"].parameters["m"].value, 3.0)
        assert np.allclose(loaded["_PartEnum.HALO"].parameters["m"].value, 7.0)


def test_to_dict_normalizes_numpy_and_scalar_subclasses():
    from collections import OrderedDict

    class _ExtraArgsPotential(NullPotential):
        _extra_serialize_args = [
            np.str_("label"),
            _StrSubclass("sub_str"),
            _PartEnum.HALO,
            "np_int",
            "np_float",
            "np_bool",
            "sub_int",
            "sub_float",
            "tup",
            "obj_arr",
            "odict",
        ]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.label = np.str_("v2")
            self.sub_str = _StrSubclass("custom")
            self.halo = _PartEnum.HALO
            self.np_int = np.int64(7)
            self.np_float = np.float32(2.5)
            self.np_bool = np.bool_(True)
            self.sub_int = _IntSubclass(3)
            self.sub_float = _FloatSubclass(1.25)
            self.tup = (np.str_("a"), np.int64(2), _FloatSubclass(0.5))
            self.obj_arr = np.array([np.str_("x"), _StrSubclass("y")], dtype=object)
            self.odict = OrderedDict([(np.str_("k"), np.int64(9))])

    pot = _ExtraArgsPotential(units=galactic)
    pot_dict = to_dict(pot)
    _assert_plain_yaml_types(pot_dict)

    extra = pot_dict["extra_args"]
    assert extra == {
        "label": "v2",
        "sub_str": "custom",
        "halo": "halo",
        "np_int": 7,
        "np_float": 2.5,
        "np_bool": True,
        "sub_int": 3,
        "sub_float": 1.25,
        "tup": ["a", 2, 0.5],
        "obj_arr": ["x", "y"],
        "odict": {"k": 9},
    }
    dumped = yaml.safe_dump(pot_dict, default_flow_style=None)
    assert "!!python/" not in dumped
    assert yaml.safe_load(dumped) == pot_dict

    class _ExtraArgsComposite(CompositePotential):
        _extra_serialize_args = [np.str_("version"), _PartEnum.HALO]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.version = np.str_("v2")
            self.halo = _PartEnum.HALO

    comp = _ExtraArgsComposite()
    comp[np.str_("disk")] = KeplerPotential(m=1e11 * u.Msun, units=galactic)
    comp_dict = to_dict(comp)
    _assert_plain_yaml_types(comp_dict)
    assert comp_dict["extra_args"] == {"version": "v2", "halo": "halo"}
    assert comp_dict["components"][0]["name"] == "disk"
    assert "!!python/" not in yaml.safe_dump(comp_dict, default_flow_style=None)

    packed = _pack_params(
        {
            np.str_("m"): 1e11 * u.Msun,
            _PartEnum.HALO: 3.0 * u.Msun,
            _StrSubclass("tag"): _PartEnum.HALO,
            "flag": np.bool_(False),
            "pair": (np.int64(1), np.float64(2.5)),
        }
    )
    _assert_plain_yaml_types(packed)
    assert packed == {
        "m": 1e11,
        "m_unit": "solMass",
        "halo": 3.0,
        "halo_unit": "solMass",
        "tag": "halo",
        "flag": False,
        "pair": [1, 2.5],
    }

    # Unexpected Quantity in extra_args must not silently lose units
    class _QuantityExtraArgsPotential(NullPotential):
        _extra_serialize_args = ["bad_qty"]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.bad_qty = 5.0 * u.kpc

    bad_pot = _QuantityExtraArgsPotential(units=galactic)
    with pytest.raises(yaml.representer.RepresenterError):
        save(bad_pot, io.StringIO())


class _DistinctStr(str):
    def __eq__(self, other):
        return type(other) is _DistinctStr and str.__eq__(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((_DistinctStr, str.__str__(self)))


@pytest.mark.parametrize("composite_cls", [CompositePotential, CCompositePotential])
def test_composite_distinct_str_duplicate_name_raises(tmp_path, composite_cls):
    distinct_halo = _DistinctStr("halo")
    assert distinct_halo != "halo"
    assert len({distinct_halo: 1, "halo": 2}) == 2

    potential = composite_cls()
    potential[distinct_halo] = KeplerPotential(m=3.0, units=DimensionlessUnitSystem())
    potential["halo"] = KeplerPotential(m=7.0, units=DimensionlessUnitSystem())

    pos = [1.0, 0.0, 0.0]
    assert np.allclose(potential.energy(pos), [-10.0])

    with pytest.raises(ValueError, match="Duplicate component name 'halo'"):
        to_dict(potential)

    with pytest.raises(ValueError, match="Duplicate component name 'halo'"):
        save(potential, tmp_path / "dup_component.yml")

    with pytest.raises(ValueError, match="Duplicate component name 'halo'"):
        save(potential, io.StringIO())


@pytest.mark.parametrize("composite_cls", [CompositePotential, CCompositePotential])
def test_composite_distinct_str_unique_names_roundtrip(tmp_path, composite_cls):
    distinct_disk = _DistinctStr("disk")
    assert distinct_disk != "disk"

    potential = composite_cls()
    potential[distinct_disk] = KeplerPotential(m=3.0, units=DimensionlessUnitSystem())
    potential["halo"] = KeplerPotential(m=7.0, units=DimensionlessUnitSystem())

    pos = [1.0, 0.0, 0.0]
    expected_energy = potential.energy(pos)
    assert np.allclose(expected_energy, [-10.0])

    pot_dict = to_dict(potential)
    _assert_plain_yaml_types(pot_dict)
    assert [c["name"] for c in pot_dict["components"]] == ["disk", "halo"]
    assert all(type(c["name"]) is str for c in pot_dict["components"])

    fn = tmp_path / "composite_distinct_str.yml"
    save(potential, fn)
    raw = fn.read_text(encoding="utf-8")
    assert "!!python/" not in raw

    buf = io.StringIO()
    save(potential, buf)
    assert buf.getvalue() == raw

    for loaded in (load(fn), load(io.StringIO(raw))):
        assert list(loaded.keys()) == ["disk", "halo"]
        assert all(type(k) is str for k in loaded)
        assert loaded["disk"] == potential[distinct_disk]
        assert loaded["halo"] == potential["halo"]
        assert np.allclose(loaded.energy(pos), [-10.0])
        assert np.allclose(loaded.energy(pos), expected_energy)
        assert np.allclose(loaded["disk"].parameters["m"].value, 3.0)
        assert np.allclose(loaded["halo"].parameters["m"].value, 7.0)


def test_duplicate_normalized_keys_raise_value_error():
    distinct_k = _DistinctStr("k")
    with pytest.raises(ValueError, match="Duplicate"):
        _to_yaml_safe({distinct_k: 1, "k": 2})

    with pytest.raises(ValueError, match="Duplicate"):
        _to_yaml_safe({"outer": {distinct_k: 1, "k": 2}})

    distinct_m = _DistinctStr("m")
    with pytest.raises(ValueError, match="Duplicate"):
        _pack_params({distinct_m: 3.0 * u.Msun, "m": 7.0 * u.Msun})

    with pytest.raises(ValueError, match="Duplicate"):
        _pack_params({"m": 3.0 * u.Msun, "m_unit": "kpc"})

    with pytest.raises(ValueError, match="Duplicate"):
        _pack_params({"m_unit": "kpc", "m": 3.0 * u.Msun})

    class _DupUnitSystem:
        def to_dict(self):
            return {_DistinctStr("length"): u.kpc, "length": u.pc}

    pot_dup_units = NullPotential(units=galactic)
    pot_dup_units._units = _DupUnitSystem()
    with pytest.raises(ValueError, match="Duplicate"):
        to_dict(pot_dup_units)

    distinct_tag = _DistinctStr("tag")

    class _DupExtraArgsPotential(NullPotential):
        _extra_serialize_args = [distinct_tag, "tag"]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__[distinct_tag] = "a"
            self.__dict__["tag"] = "b"

    with pytest.raises(ValueError, match="Duplicate"):
        to_dict(_DupExtraArgsPotential(units=galactic))

    distinct_ver = _DistinctStr("version")

    class _DupExtraArgsComposite(CompositePotential):
        _extra_serialize_args = [distinct_ver, "version"]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__[distinct_ver] = "v1"
            self.__dict__["version"] = "v2"

    comp = _DupExtraArgsComposite()
    comp["halo"] = KeplerPotential(m=1e11 * u.Msun, units=galactic)
    with pytest.raises(ValueError, match="Duplicate"):
        to_dict(comp)
