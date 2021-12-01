import pytest
import torch

from boip.initialize import initialize

@pytest.fixture(params=[1000, 5000, 10000])
def n(request):
    return request.param

@pytest.fixture(params=[2, 4, 8])
def d(request):
    return request.param

@pytest.fixture
def choices(n, d):
    return torch.rand((n, d))

@pytest.fixture(params=[10, 50, 100, 200])
def m(request):
    return request.param

@pytest.fixture(params=[0, 14, 42])
def seed(request):
    return request.param

@pytest.mark.parametrize('_', range(3))
def test_initialize_no_seed(choices, m, _):
    X1 = initialize(None, m, choices, None)
    X2 = initialize(None, m, choices, None)

    assert not torch.equal(X1, X2)

def test_initialize_seeded(choices, m, seed):
    X1 = initialize(None, m, choices, seed)
    X2 = initialize(None, m, choices, seed)

    assert torch.equal(X1, X2)