from typing import Optional

from scipy.stats import qmc
import torch
from torch import Tensor

from enum import auto


from boip.utils import AutoName


class InitMode(AutoName):
    LHC = auto()
    UNIFORM = auto()


def initialize(
    N: int, choices: Tensor, seed: Optional[int] = None, init_mode: InitMode = InitMode.UNIFORM
) -> Tensor:
    """select N points from choices with the given seed and mode and return their indices

    Parameters
    ----------
    N : int
        the number of selections to make
    chocies : Tensor
        the points to select from
    seed : Optiona[int], default=None
        the initalization seed
    init_mode : InitMode, default=InitMode.UNIFORM
        the initalization mode:
        - InitMode.UNIFORM: select N points at random
        - InitMode.LHC: use latin hypercube sampling to select N initial points from within the
        bounds of the design space specfied by `choices` then select the nearest neighbors within
        the design space

    Returns
    -------
    idxs : Tensor
        the indices of the points selected in `choices`
    """
    if init_mode == InitMode.UNIFORM:
        return init_uniform(N, choices, seed)
    elif init_mode == InitMode.LHC:
        return init_LHC(N, choices, seed)
    raise ValueError(f"Invalid init_mode! got: {init_mode}")


def init_uniform(N: int, choices: Tensor, seed: Optional[int] = None):
    with torch.random.fork_rng(range(torch.cuda.device_count())):
        if seed is not None:
            torch.random.manual_seed(seed)
        else:
            torch.random.seed()

        idxs = torch.randperm(len(choices))[:N]

    return idxs


def init_LHC(N: int, choices: Tensor, seed: Optional[int] = None):
    sampler = qmc.LatinHypercube(choices.shape[1], seed=seed)

    l_bounds = choices.min(0)[0]
    u_bounds = choices.max(0)[0]

    X = sampler.random(N)
    X = qmc.scale(X, l_bounds, u_bounds)
    X = torch.as_tensor(X, dtype=choices.dtype, device=choices.device)

    return torch.cdist(X, choices).argmin(dim=1)
