import pytest

from ..util import ImmutableDict


def test_immutabledict():
    a = {"a": 5, "c": 6}
    b = ImmutableDict(**a)

    with pytest.raises(TypeError):
        b["test"] = 5
