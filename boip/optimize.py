from typing import Optional, Tuple

from botorch.acquisition import UpperConfidenceBound
from botorch.test_functions.base import BaseTestProblem
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
from tqdm import tqdm

from boip.initialize import initialize
from boip.prune import prune
from boip.train import fit_model

def optimize(
    obj: BaseTestProblem,
    N: int,
    T: int,
    choices: Tensor,
    q: int = 1,
    prune_inputs: bool = False,
    k: int = 1,
    prob: float = 0.,
    verbose: bool = False,
    init_seed: Optional[int] = None
) -> Tuple[Tensor, Tensor]:
    """Optimize the input objective

    Parameters
    ----------
    obj : BaseTestProblem
        the objective function to optimize
    N : int
        the number of initialization points
    T : int
        the number of iterations to optimize over
    choices: Tensor
        the discrete choices
    q : int, default=1
        the number of points to take in each batch
    prune : bool, default=False
        whether to prune the input space irreversibly
    k : int, default=1
        the rank of point to compare each candidate pruning point to
    prob : float, default=0.
        the mimimum probability a candidate point must have to improve upon the k-th best
        predicted mean in order to be retained
    verbose : bool, default=False
        whether to print
    init_seed: Optional[int] = None
        the seed with which to sample random initial points

    Returns
    -------
    X : Tensor
        a `q*(T+1) x d` tensor of the acquired input points
    Y : Tensor
        a `q*(T+1) x 1` tensor of the associated objective values
    S : np.ndarray
        a `(T+1)` length vector containing the size of the input space at each iteration
    """
    S = np.empty(T+1)
    S[0] = len(choices)

    idxs = initialize(obj, N, choices, init_seed)
    X = choices[idxs]
    Y = obj(X).reshape(-1, 1)

    mask = torch.ones(len(choices), dtype=bool)
    mask[idxs] = False
    choices = choices[mask]

    for t in tqdm(range(T), "Optimizing", disable=not verbose):
        model = SingleTaskGP(X, (Y - Y.mean(0)) / Y.std(0))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        optim = Adam(model.parameters(), lr=0.001)
        fit_model(X, model, optim, mll, verbose=verbose)

        if prune_inputs:
            idxs, E_opt = prune(choices, model, k, prob)
            if len(idxs) == len(choices):
                if verbose:
                    print("Did not prune pool!")
            else:
                #NOTE(degraff): could implement something here about fast recovery
                choices = choices[idxs]
                if verbose:
                    print(f"Pruned pool to {len(idxs)} choices!")
                    print(f"Expected optima pruned: {E_opt:0.3f}")

        S[t+1] = len(choices) + q*(t+1)

        acqf = UpperConfidenceBound(model, beta=2)
        A = acqf(torch.unsqueeze(choices, 1))
        
        _, idxs = torch.topk(A, q, dim=0, sorted=True)
        X_t = choices[idxs]
        Y_t = obj(X_t).reshape(-1, 1)

        mask = torch.ones(len(choices), dtype=bool)
        mask[idxs] = False
        choices = choices[mask]

        X = torch.cat((X, X_t))
        Y = torch.cat((Y, Y_t))

        if len(choices) == 0:
            break

    return X, Y, S
