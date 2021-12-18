from typing import Tuple

import torch
from torch import Tensor
from botorch.models.model import Model

def prune(
    choices: Tensor,
    model: Model,
    k: int,
    prob: float,
    mask: Tensor,
    alpha: float = 1.,
) -> Tuple[Tensor, Tensor]:
    """prune the possible choices based on current model's beliefs

    Parameters
    ----------
    choices : Tensor
        The enumerated choices in the design space
    model : Model
        a trained model
    k : int
        the absolute rank which constitutes a hit
    prob : float
        the minimum probability of being a hit below which a point will be pruned
    mask : Tensor
        the choices in the pool that have already been pruned or acquired
    alpha : float, default=1.
        the amount by which to scale the uncertainty

    Returns
    -------
    idxs : Tensor
        the indices to prune
    E_opt : Tensor
        the total expected number of optima pruned
    """
    model.eval()

    with torch.no_grad():
        Y_hat = model(choices)

    return pruned_idxs_prob(Y_hat.mean, alpha*Y_hat.variance, k, prob, mask)

def pruned_idxs_prob(Y_mean, Y_var, k, prob, mask) -> Tuple[Tensor, Tensor]:
    threshold = torch.topk(Y_mean, k, dim=0, sorted=True)[0][-1]
    P = prob_above(Y_mean, Y_var, threshold)

    idxs = torch.arange(len(Y_mean))[~mask]
    P = P[~mask]

    idxs_to_prune = P < prob

    return idxs[idxs_to_prune], P[idxs_to_prune].sum()

def prob_above(Y_mean: Tensor, Y_var: Tensor, threshold: Tensor) -> Tensor:
    """the probability that each prediction (given mean and uncertainty) is above the input 
    threshold"""
    Z = (Y_mean - threshold) / Y_var.sqrt()
    return torch.distributions.normal.Normal(0, 1).cdf(Z)

def thompson(Y_post, n: int):
    Y_sample = Y_post.sample(torch.empty(n).shape)
    return