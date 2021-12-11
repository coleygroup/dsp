from typing import Tuple

import torch
from torch import Tensor

@torch.inference_mode()
def prune(choices: Tensor, model, k, prob) -> Tuple[Tensor, Tensor]:
    Y_hat = model.posterior(choices)

    return retained_idxs(Y_hat.mean, Y_hat.variance, k, prob)
    # threshold = torch.topk(Y_hat.mean, k, dim=0, sorted=True)[0][-1]
    # P = prob_above(Y_hat.mean, Y_hat.variance, threshold).sum(1)
    # idxs = torch.arange(len(Y_hat.mean))[P >= prob]

    # return idxs, P[P < prob].sum()

def retained_idxs(Y_mean, Y_var, k, prob) -> Tuple[Tensor, Tensor]:
    threshold = torch.topk(Y_mean, k, dim=0, sorted=True)[0][-1]
    P = prob_above(Y_mean, Y_var, threshold).sum(1)
    idxs = torch.arange(len(Y_mean))[P >= prob]

    return idxs, P[P < prob].sum()

def prob_above(Y_mean: Tensor, Y_var: Tensor, threshold: Tensor) -> Tensor:
    """the probability that each prediction (given mean and uncertainty) is above the input 
    threshold"""
    Z = (Y_mean - threshold) / Y_var.sqrt()
    return torch.distributions.normal.Normal(0, 1).cdf(Z)