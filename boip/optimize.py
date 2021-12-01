from timeit import default_timer as time
from typing import Optional, Tuple

from botorch.acquisition import UpperConfidenceBound, qUpperConfidenceBound
from botorch.test_functions.base import BaseTestProblem
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf_discrete
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam

from boip.initialize import initialize
from boip.prune import prune
from boip.train import fit_model
from boip.window import window_size

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
        an `(N+T) x d` tensor of the acquired input points
    Y : Tensor
        an `(N+T) x 1` tensor of the associated objective values
    S : np.ndarray
        an `(N+T)` length vector containing the size of the input space at each iteration
    dT : np.ndarray
        an `(N+T)` length vector containing the time elapsed from the start of optimization at the
        given iteration. The first N entries are always 0
    """
    # idxs = torch.randperm(len(choices))[:N]
    # X = choices[idxs]
    X = initialize(obj, N, choices, init_seed)
    Y = obj(X).reshape(-1, 1)

    S = np.empty(N+T)
    S[:] = len(choices)

    dT = np.empty(N+T)
    start = time()

    u = 0
    t = 0

    while t < T:
        model = SingleTaskGP(X, (Y - Y.mean(0)) / Y.std(0))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        optim = Adam(model.parameters(), lr=0.001)
        fit_model(X, model, optim, mll)

        if prune_inputs:
            idxs, E_opt = prune(choices, model, k, prob)
            if len(idxs) == len(choices):
                u += 1
                if verbose:
                    print("Did not prune pool!")
            else:
                #NOTE(degraff): could implement something here about fast recovery
                u = 0
                choices = choices[idxs]
                if verbose:
                    print(f"Pruned pool to {len(idxs)} choices!")
                    print(f"Expected optima pruned: {E_opt}")
            q = min(window_size(u), T-t)
        else:
            q = 1

        acqf = UpperConfidenceBound(model, beta=2)
        A = acqf(torch.unsqueeze(choices, 1))
        
        X_t = choices[torch.topk(A, q, dim=0, sorted=True)[1]]
        Y_t = obj(X_t).reshape(-1, 1)

        X = torch.cat((X, X_t))
        Y = torch.cat((Y, Y_t))

        S[N+t:N+t+q] = len(choices)
        dT[N+t:N+t+q] = time()

        t += q

    dT[:N] = 0
    dT[N:] -= start

    return X, Y, S, dT
