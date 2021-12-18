from typing import Optional

from botorch.test_functions.base import BaseTestProblem
import torch
from torch import Tensor


def discretize(obj: BaseTestProblem, N: int, seed: Optional[int] = None) -> Tensor:
    """generate a discretized landscape of the input objective

    Parameters
    ----------
    obj : BaseTestProblem
        the objective to discretize
    N : int
        the number of samples with which to discretize the landscape
    seed : Optional[int], default=None
        the random seed. If None, use a randomly generated random seed.
        NOTE: this function forks the RNG state.

    Returns
    -------
    choices : Tensor
        an `n x d` tensor containing the discrete choices, where `n` is the number of choices and
        `d` is the dimension of the input domain
    """
    with torch.random.fork_rng(range(torch.cuda.device_count())):
        if seed is not None:
            torch.random.manual_seed(seed)
        else:
            torch.random.seed()

        choices = torch.distributions.uniform.Uniform(*obj.bounds).sample([N])

    return choices
