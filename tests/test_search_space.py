import pytest
import numpy as np
from hps import search_space, Real, Integer, Categorical
from numbers import Number


@pytest.mark.parametrize("dimension", [
    Real("Real", 1e-4, 1e-1),
    Integer("Integer", 10, 100),
    Categorical("Categorical", ["cat1", "cat2", "cat3"]),
])
def test_to_numeric(dimension):
    assert isinstance(dimension.to_numeric(dimension.get_rnd()), Number)


@pytest.mark.parametrize("dimension", [
    Real("Real", 1e-4, 1e-1),
    Integer("Integer", 10, 100),
    Categorical("Categorical", ["cat1", "cat2", "cat3"]),
])
def test_from_numeric(dimension):
    rn_val = dimension.get_rnd()
    if isinstance(rn_val, float):
        np.testing.assert_allclose(dimension.from_numeric(dimension.to_numeric(rn_val)), rn_val)
    else:
        assert dimension.from_numeric(dimension.to_numeric(rn_val)) == rn_val



