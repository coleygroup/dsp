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
    prob: float = 0.025,
    alpha: float = 1.0,
    no_reacquire: bool = True,
    verbose: int = 0,
    init_seed: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
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
        an `n x d` tensor containing the choices over which to optimize
    q : int, default=1
        the number of points to take in each batch
    prune : bool, default=False
        whether to prune the input space irreversibly
    k : int, default=1
        the rank of point to compare each candidate pruning point to
    prob : float, default=0.
        the mimimum probability a candidate point must have to improve upon the k-th best
        predicted mean in order to be retained
    alpha : float, default=1.0
        the amount by which to scale the uncertainties
    no_reacquire : bool, default=True
        whether points can be reacquired
    verbose : int, default=0
        the amount of information to print
    init_seed: Optional[int] = None
        the seed with which to sample random initial points

    Returns
    -------
    X : Tensor
        a `q*(T+1) x d` tensor of the acquired input points
    Y : Tensor
        a `q*(T+1) x 1` tensor of the associated objective values
    H : Tensor
        an `n x 2` tensor containing the iteration at which each point was either acquired or
        pruned, where n is the number of choices in the pool. The 0th column indicates the
        iteration at which the point was acquired and the 1st column indicates the iteration at
        which the point was pruned. A value of -1 indicates that the point was neither acquired
        or pruned, respectively.
        NOTE: points may only be acquired OR pruned, they may not be both.
        NOTE: if points are allowed to be reacquired, then the value in column 0 indicates the
        latest iteration at which the point was acquired, as the previous value will be overwritten
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    choices = choices.to(device)

    acq_idxs = initialize(obj, N, choices, init_seed)
    X = choices[acq_idxs]
    Y = obj(X).unsqueeze(1)

    acq_mask = torch.zeros(len(choices)).bool().to(device)
    acq_mask[acq_idxs] = True

    prune_mask = torch.zeros(len(choices)).bool().to(device)

    H = torch.zeros((len(choices), 2)).long().to(device) - 1
    H[acq_idxs, 0] = 0

    for t in tqdm(range(1, T + 1), "Optimizing", disable=verbose < 1):
        model = SingleTaskGP(X, (Y - Y.mean(0)) / Y.std(0))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        optim = Adam(model.parameters(), lr=0.001)
        fit_model(X, model, optim, mll, verbose=verbose)

        if prune_inputs:
            pruned_idxs, _ = prune(choices, model, k, prob, prune_mask + acq_mask, alpha)

            prune_mask[pruned_idxs] = True
            H[pruned_idxs, 1] = t

            # if len(pruned_idxs) == 0:
            #     if verbose:
            #         print("Did not prune pool!")
            # else:
            #     if verbose:
            #         print(f"Pruned {len(pruned_idxs)} choices!")
            #         print(f"Expected optima pruned: {E_opt:0.3f}")

        acqf = UpperConfidenceBound(model, beta=2)
        A = acqf(choices.unsqueeze(1))
        A[prune_mask] = -np.inf
        if no_reacquire:
            A[acq_mask] = -np.inf

        _, acq_idxs = torch.topk(A, q, dim=0, sorted=True)
        X_t = choices[acq_idxs]
        Y_t = obj(X_t).unsqueeze(1)

        acq_mask[acq_idxs] = True
        H[acq_idxs, 0] = t

        X = torch.cat((X, X_t))
        Y = torch.cat((Y, Y_t))

        if no_reacquire and len(choices[~(acq_mask + prune_mask)]) == 0:
            print("no points left! Stopping...")
            break
        elif len(choices[~prune_mask]) == 0:
            print("no points left! Stopping...")
            break

    return X, Y, H
