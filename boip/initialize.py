from typing import Optional

from botorch.test_functions.base import BaseTestProblem
import torch
from torch import Tensor


def initialize(
    obj: BaseTestProblem,
    N: int,
    choices: Tensor,
    seed: Optional[int] = None
) -> Tensor:
    with torch.random.fork_rng(range(torch.cuda.device_count())):
        if seed is not None:
            torch.random.manual_seed(seed)
        else:
            torch.random.seed()

        idxs = torch.randperm(len(choices))[:N]
    
    return choices[idxs]