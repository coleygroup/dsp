from copy import deepcopy
import pytest

from botorch.acquisition import UpperConfidenceBound
from botorch.test_functions.base import BaseTestProblem
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch
from torch.optim import Adam

from dsp.train import fit_model


@pytest.fixture(params=[10, 50, 100])
def N(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def M(request):
    return request.param


@pytest.fixture()
def X(N, M):
    return torch.rand(N, M)


@pytest.fixture
def Y(N):
    return torch.rand(N, 1)


def test_fit_model(X, Y):
    model = SingleTaskGP(X, (Y - Y.mean(0)) / Y.std(0))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    optim = Adam(model.parameters(), lr=0.001)

    params_orig = deepcopy(list(model.parameters()))
    fit_model(X, model, optim, mll, epochs=5)
    params_new = deepcopy(list(model.parameters()))

    assert not all(torch.equal(p1, p2) for p1, p2 in zip(params_orig, params_new))


def test_fit_model_no_lr(X, Y):
    model = SingleTaskGP(X, (Y - Y.mean(0)) / Y.std(0))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    optim = Adam(model.parameters(), lr=0.0)

    params_orig = deepcopy(list(model.parameters()))
    fit_model(X, model, optim, mll, epochs=5)
    params_new = deepcopy(list(model.parameters()))

    assert all(torch.equal(p1, p2) for p1, p2 in zip(params_orig, params_new))
