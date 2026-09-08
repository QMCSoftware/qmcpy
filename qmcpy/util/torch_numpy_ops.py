import numpy as np


def get_npt(x):
    """Return the array backend module matching the input.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Array whose backend is wanted.

    Returns:
        module: ``numpy`` for an ``np.ndarray``, otherwise ``torch``.

    Raises:
        AssertionError: If ``x`` is neither an ``np.ndarray`` nor a ``torch.Tensor``.
    """
    if isinstance(x, np.ndarray):
        return np
    else:
        import torch

        if not (isinstance(x, torch.Tensor)):
            raise AssertionError
        return torch
