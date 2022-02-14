import pytest
import torch

from boip.prune import pruned_idxs_prob, prob_above

@pytest.fixture(params=[100*i+1 for i in range(4)])
def N(request):
    return request.param

@pytest.fixture
def Y_mean(N):
    return torch.distributions.Uniform(0, 1).sample([N])

@pytest.fixture
def Y_var(Y_mean):
    return 0.1 * torch.rand_like(Y_mean)

@pytest.fixture
def Y_var_tiny(Y_mean):
    return 1e-6 * torch.ones_like(Y_mean)

@pytest.fixture(params=[1., 0.1, 0.])
def prob(request):
    return torch.tensor(request.param) 

def test_pruned_idxs(Y_mean: torch.Tensor, Y_var, prob):
    threshold = torch.topk(Y_mean, 1, dim=0, sorted=True)[0][-1]
    P = prob_above(Y_mean, Y_var, threshold)

    idxs, E = pruned_idxs_prob(Y_mean, Y_var, threshold, prob, torch.zeros(len(Y_mean), dtype=bool))

    assert (P[idxs] < prob).all()
    torch.testing.assert_close(P[P < prob].sum(), E)

def test_retain_all(Y_mean, Y_var_tiny):
    threshold = torch.topk(Y_mean, 1, dim=0, sorted=True)[0][-1]
    idxs, _ = pruned_idxs_prob(Y_mean, Y_var_tiny, threshold, 0., torch.zeros(len(Y_mean), dtype=bool))

    torch.testing.assert_close(torch.arange(0), idxs, rtol=0, atol=0)

def test_retain_none(Y_mean, Y_var_tiny):
    threshold = torch.topk(Y_mean, 1, dim=0, sorted=True)[0][-1]
    idxs, _ = pruned_idxs_prob(Y_mean, Y_var_tiny, threshold, 1.01, torch.zeros(len(Y_mean), dtype=bool))

    torch.testing.assert_close(torch.arange(len(Y_mean)), idxs, rtol=0, atol=0)

@pytest.mark.parametrize(
    "threshold,expected_prob", [(-1, 1), (2, 0)]
)
def test_prob_above_tiny_var(Y_mean, Y_var_tiny, threshold, expected_prob):
    P = prob_above(Y_mean, Y_var_tiny, torch.tensor(threshold))

    torch.testing.assert_close(P, expected_prob * torch.ones_like(P))
