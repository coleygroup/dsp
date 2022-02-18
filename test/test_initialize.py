import pytest
import torch

from boip.initialize import InitMode, initialize


@pytest.fixture(params=[100, 500, 1000])
def n(request):
    return request.param


@pytest.fixture(params=[2, 4, 8])
def d(request):
    return request.param


@pytest.fixture
def choices(n, d):
    return torch.rand((n, d))


@pytest.fixture(params=[10, 20, 50])
def m(request):
    return request.param


@pytest.fixture(params=[0, 14, 42])
def seed(request):
    return request.param


def test_inavlid_init_mode():
    with pytest.raises(ValueError):
        initialize(0, [], None, "foo")


@pytest.mark.parametrize("_", range(3))
def test_init_uniform_no_seed(choices, m, _):
    I = initialize(m, choices, None, InitMode.UNIFORM)
    J = initialize(m, choices, None, InitMode.UNIFORM)

    assert not torch.equal(I, J)


def test_init_uniform_seeded(choices, m, seed):
    I = initialize(m, choices, seed, InitMode.UNIFORM)
    J = initialize(m, choices, seed, InitMode.UNIFORM)

    assert torch.equal(I, J)


@pytest.mark.parametrize("_", range(3))
def test_init_LHC_no_seed(choices, m, _):
    I = initialize(m, choices, None, InitMode.LHC)
    J = initialize(m, choices, None, InitMode.LHC)

    assert not torch.equal(I, J)


def test_init_LHC_seeded(choices, m, seed):
    I = initialize(m, choices, seed, InitMode.LHC)
    J = initialize(m, choices, seed, InitMode.LHC)

    assert torch.equal(I, J)
