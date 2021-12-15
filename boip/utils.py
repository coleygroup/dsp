import torch
from torch import Tensor

def unselect(X: Tensor, idxs: Tensor) -> Tensor:
    mask = torch.ones(len(X)).bool()
    mask[idxs] = False

    return X[mask]