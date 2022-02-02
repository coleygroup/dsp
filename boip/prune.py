from typing import Optional, Tuple, Union
from numpy import isin

import torch
from torch import Tensor
from botorch.models.model import Model


def prune(
    choices: Tensor,
    model: Model,
    k_or_threshold: Union[int, float],
    prob: float,
    mask: Tensor,
    threshold: Optional[float] = None,
    gamma: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """prune the possible choices based on current model's beliefs

    Parameters
    ----------
    choices : Tensor
        The enumerated choices in the design space
    model : Model
        a trained model
    k_or_threshold : Union[int, float]
        either the rank which constitutes a hit or the threshold that defines one.
        I.e, the top-k predictions are predicted hits or points with a predicted mean above the
        threshold are defined as hits.
    prob : float
        the minimum probability of being a hit below which a point will be pruned
    mask : Tensor
        the choices in the pool that have already been pruned or acquired
    threshold : Optional[float], default=None
        the threshold to use for pruning. If None, use the k-th best predicted mean
    gamma : float, default=1.0
        the amount by which to scale the variance

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

    if isinstance(k_or_threshold, int):
        k = k_or_threshold
        threshold = torch.topk(Y_hat.mean, k, dim=0, sorted=True)[0][-1]
    elif isinstance(k_or_threshold, float):
        threshold = torch.tensor(k_or_threshold)
    else:
        raise TypeError(
            f"k_or_threshold must be of type [int, float]! got: {type(k_or_threshold)}"
        )

    return pruned_idxs_prob(Y_hat.mean, gamma * Y_hat.variance, threshold, prob, mask)


def pruned_idxs_prob(
    Y_mean: Tensor, Y_var: Tensor, threshold: Tensor, prob: float, mask: Tensor
) -> Tuple[Tensor, Tensor]:
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
