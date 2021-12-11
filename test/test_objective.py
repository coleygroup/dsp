import pytest
import uuid

import torch

from boip import objectives

@pytest.fixture(params=objectives.valid_objectives())
def obj(request):
    return objectives.build_objective(request.param)

@pytest.fixture(params=[str(uuid.uuid4()) for _ in range(3)])
def uuid(request):
    return request.param

@pytest.fixture(params=[10000, 50000, 100000])
def N(request):
    return request.param

@pytest.fixture(params=[0, 15, 29])
def seed(request):
    return request.param

def test_invalid_objective(uuid):
    with pytest.raises(ValueError):
        objectives.build_objective(uuid)

def test_discretize_empty(obj, seed):
    choices1 = objectives.discretize(obj, 0, seed)

    assert choices1.nelement() == 0

@pytest.mark.parametrize('_', range(3))
def test_discretize_randomness(obj, N, _):
    choices1 = objectives.discretize(obj, N, None)
    choices2 = objectives.discretize(obj, N, None)

    assert not torch.equal(choices1, choices2)

def test_discretize_seeded(obj, N, seed):
    choices1 = objectives.discretize(obj, N, seed)
    choices2 = objectives.discretize(obj, N, seed)

    assert torch.equal(choices1, choices2)