from timeit import default_timer as time
from typing import Tuple

from botorch.acquisition import UpperConfidenceBound
from botorch.test_functions.base import BaseTestProblem
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf_discrete
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
from tqdm import trange

from boip.prune import prune
from boip.train import fit_model

NUM_RESTARTS = 20
RAW_SAMPLES = 20

def optimize(
    obj: BaseTestProblem,
    N: int,
    T: int,
    choices: Tensor,
    prune_inputs: bool = False,
    k: int = 1,
    prob: float = 0.,
    verbose: bool = False
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
    prune : bool, default=False
        whether to prune the input space irreversibly

    Returns
    -------
    X : Tensor
        an `(N+T) x d` tensor of the acquired input points
    Y : Tensor
        an `(N+T) x 1` tensor of the associated objective values
    S : np.ndarray
        an `(N+T)` length vector containing the size of the solution space at each iteration
    S : np.ndarray
        an `(N+T)` length vector containing the time elapsed from the start of optimization at the
        given iteration. The first N entries are always 0
    """
    idxs = torch.randperm(len(choices))[:N]
    X = choices[idxs]
    Y = obj(X).reshape(-1, 1)

    S = np.empty(N+T)
    S[:] = len(choices)

    dT = np.empty(N+T)
    dT[:N] = time()

    for t in trange(T, desc="Optimizing", leave=False, disable=not verbose):
        model = SingleTaskGP(X, (Y - Y.mean(dim=0)) / Y.std(dim=0))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        optim = Adam(model.parameters(), lr=0.001)
        fit_model(X, model, optim, mll)

        if prune_inputs:
            idxs, E_opt = prune(choices, model, k, prob)
            choices = choices[idxs]
            if verbose:
                print(f"Pruned pool to {len(idxs)} choices!")
                print(f"Expected optima pruned: {E_opt}")
        S[N+t] = len(choices)
        dT[N+t] = time()
        acqf = UpperConfidenceBound(model, beta=2)
        X_t, _ = optimize_acqf_discrete(acqf, 1, choices)

        # if choices is None:
        #     X_t, _ = optimize_acqf(acqf, obj.bounds, 1, NUM_RESTARTS, RAW_SAMPLES)
        # else:


        Y_t = obj(X_t).reshape(-1, 1)

        X = torch.cat((X, X_t))
        Y = torch.cat((Y, Y_t))

    dT[N:] -= dT[0]
    return X, Y, S, dT
