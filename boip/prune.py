import torch
from torch import Tensor

def prune(choices: Tensor, model, k, prob) -> Tensor:
    with torch.no_grad():
        Y_hat = model.posterior(choices)

    # import pdb; pdb.set_trace()
    threshold = torch.topk(Y_hat.mean, k, dim=0, sorted=True)[0][-1]
    P = prob_above(Y_hat.mean, Y_hat.variance, threshold).sum(1)
    idxs = torch.arange(len(Y_hat.mean))[P >= prob]

    return idxs, P[P < prob].sum()

def prob_above(Y_mean: Tensor, Y_var: Tensor, threshold: Tensor) -> Tensor:
    """the probability that each prediction (given mean and uncertainty) is above the input 
    threshold"""
    Z = (Y_mean - threshold) / Y_var.sqrt()
    return torch.distributions.normal.Normal(0, 1).cdf(Z)