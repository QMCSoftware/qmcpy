import numpy as np


def get_npt(x):
    if isinstance(x, np.ndarray):
        return np
    else:
        import torch

        if not (isinstance(x, torch.Tensor)):
            raise AssertionError
        return torch
