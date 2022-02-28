import math
from typing import List, Optional, Tuple
from botorch.test_functions.synthetic import SyntheticTestFunction

import torch
from torch import Tensor


class GSobol(SyntheticTestFunction):
    """Cosines test function"""

    dim = 2
    _bounds = [(0, 5) for _ in range(2)]
    _optimal_value = 0.8
    _optimizers = [tuple(0.0 for _ in range(8))]

    def evaluate_true(self, X: Tensor) -> Tensor:
        return 1 - (self.g(X) - self.r(X)).sum(dim=1)

    def g(self, X: Tensor) -> Tensor:
        return (1.6 * X - 0.5) ** 2

    def r(self, X: Tensor) -> Tensor:
        return 0.3 * torch.cos(3 * math.pi * (1.6 * X - 0.5))
